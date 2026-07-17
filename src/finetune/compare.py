"""Сравнение base-модели и LoRA fine-tuned: 12 вопросов, метрики стиля, отчёт.

Генерирует ответы base (Qwen/Qwen2.5-1.5B-Instruct) и base+adapter
(models/lora_adapter) на 12 вопросов вне датасета (5 IT, 5 HR, 2 small-talk),
считает метрики стиля (доля ответов со «Здравствуйте», LLM-as-judge 1–5 через
YandexGPT), пишет reports/finetune_examples.json и reports/finetune_report.md.
Scores base_style_score/ft_style_score отправляются в Langfuse (best-effort).

Модели грузятся строго последовательно (base → del/gc → base+adapter): RAM
ограничена. Запуск: .venv/bin/python -m src.finetune.compare
"""

import gc
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import get_settings
from src.observability.tracing import (
    init_langfuse,
    langfuse_scope,
    suppress_langfuse_errors,
)
from src.rag.chain import YandexChatOpenAI

SYSTEM_PROMPT = (
    "Ты — внутренний ассистент компании «ТехноСфера». Отвечай вежливо, "
    "структурированно, в корпоративном стиле. Опирайся только на корпоративную "
    "базу знаний; если ответа в ней нет — честно скажи об этом и предложи "
    "контакт поддержки."
)

# 12 вопросов ВНЕ dataset.jsonl: 5 IT, 5 HR, 2 small-talk.
QUESTIONS = [
    ("IT", "VPN подключён, но корпоративные сайты не открываются. Что проверить?"),
    ("IT", "Как сбросить пароль от учётной записи, если забыл текущий?"),
    ("IT", "Принтер печатает полосами, куда обращаться?"),
    ("IT", "Как создать новый репозиторий в корпоративном GitLab?"),
    ("IT", "Почта на телефоне перестала синхронизироваться, что делать?"),
    ("HR", "Как перенести неиспользованные дни отпуска на следующий год?"),
    ("HR", "Где посмотреть, сколько дней отпуска у меня осталось?"),
    ("HR", "Как получить справку о доходах для банка?"),
    ("HR", "Когда проходит performance review и к чему готовиться?"),
    ("HR", "Можно ли оформить ДМС для супруга за счёт компании?"),
    ("small-talk", "Привет! Как у тебя дела?"),
    ("small-talk", "Что ты любишь делать в свободное время?"),
]

MAX_NEW_TOKENS = 220

JUDGE_PROMPT = """Оцени соответствие ответа ассистента корпоративному стилю по шкале от 1 до 5.

Критерии: вежливость, деловой тон, структурированность, приветствие «Здравствуйте», отсутствие фамильярности и лишней болтовни.
5 — идеальный корпоративный стиль, 1 — совершенно не соответствует.

Вопрос пользователя: {question}
Ответ ассистента: {answer}

Верни ТОЛЬКО одно целое число от 1 до 5, без пояснений."""


def generate_answers(model, tokenizer, questions) -> list[str]:
    """Greedy-генерация ответов на вопросы с chat template и system-промптом."""
    answers = []
    for category, question in questions:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"  [{category}] {question[:50]}... done", flush=True)
    return answers


def judge_answers(llm, questions, answers) -> list[int | None]:
    """LLM-as-judge: оценка 1–5 «соответствие корпоративному стилю»."""
    scores = []
    for (category, question), answer in zip(questions, answers):
        try:
            response = llm.invoke(
                JUDGE_PROMPT.format(question=question, answer=answer)
            )
            match = re.search(r"[1-5]", response.content)
            scores.append(int(match.group()) if match else None)
        except Exception as exc:  # сеть/лимиты — пропускаем ответ
            print(f"  judge error: {exc}", flush=True)
            scores.append(None)
    return scores


def _mean(values: list[int | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 2) if present else None


def _hello_share(answers: list[str]) -> float:
    return round(
        sum(1 for a in answers if a.startswith("Здравствуйте")) / len(answers), 2
    )


def _parse_loss_log(log_path: Path) -> list[dict]:
    """Строки {'loss': ..., 'epoch': ...} из лога обучения."""
    entries = []
    if not log_path.exists():
        return entries
    for match in re.finditer(
        r"\{'loss': '([\d.]+)'.*? 'epoch': '([\d.]+)'\}", log_path.read_text()
    ):
        entries.append({"loss": float(match.group(1)), "epoch": float(match.group(2))})
    return entries


def _cell(text: str, limit: int = 400) -> str:
    text = text.replace("|", "\\|").replace("\n", "<br>")
    return text[:limit] + ("…" if len(text) > limit else "")


def render_report(
    config: dict,
    loss_entries: list[dict],
    examples: list[dict],
    metrics: dict,
) -> str:
    lines = [
        "# Отчёт: LoRA fine-tuning Qwen2.5-1.5B — Base vs Fine-Tuned",
        "",
        "## Конфигурация обучения",
        "",
        f"- Базовая модель: `{config['base_model']}` (CPU, {config['dtype']})",
        f"- LoRA: r={config['lora_r']}, alpha={config['lora_alpha']}, "
        f"dropout={config['lora_dropout']}, target_modules={config['lora_targets']}",
        f"- Гиперпараметры: batch={config['batch_size']}, "
        f"grad_accum={config['grad_accum']}, epochs={config['epochs']}, "
        f"lr={config['lr']}, max_len={config['max_len']}, seed={config['seed']}",
        f"- Датасет: `{config['dataset']}` ({config['dataset_size']} примеров)",
        "",
        "## Динамика loss",
        "",
        "| Эпоха | Loss |",
        "|---|---|",
    ]
    for entry in loss_entries:
        lines.append(f"| {entry['epoch']:.2f} | {entry['loss']:.4g} |")
    lines += [
        "",
        f"Начальный loss: **{loss_entries[0]['loss']}** → финальный: "
        f"**{loss_entries[-1]['loss']}** "
        f"(train_loss по итогам: {config['train_loss']})."
        if loss_entries
        else "",
        "",
        "## Метрики стиля",
        "",
        "| Метрика | Base | Fine-Tuned |",
        "|---|---|---|",
        f"| Доля ответов со «Здравствуйте» | {metrics['base_hello']:.0%} | "
        f"{metrics['ft_hello']:.0%} |",
        f"| LLM-as-judge (1–5, корп. стиль), n={metrics['judge_n']} | "
        f"{metrics['base_judge']} | {metrics['ft_judge']} |",
        "",
        "## Примеры ответов",
        "",
        "| # | Категория | Вопрос | Base | Fine-Tuned |",
        "|---|---|---|---|---|",
    ]
    for i, ex in enumerate(examples, 1):
        lines.append(
            f"| {i} | {ex['category']} | {_cell(ex['question'], 120)} | "
            f"{_cell(ex['base_answer'])} | {_cell(ex['ft_answer'])} |"
        )
    lines += ["", "## Выводы", "", config["conclusions"], ""]
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    base_model = settings.finetune_base_model
    adapter_dir = "models/lora_adapter"
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print("=== Base model ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    base_answers = generate_answers(model, tokenizer, QUESTIONS)
    del model
    gc.collect()

    print("=== Fine-tuned (base + LoRA adapter) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter_dir)
    ft_answers = generate_answers(model, tokenizer, QUESTIONS)
    del model
    gc.collect()

    print("=== LLM-as-judge ===", flush=True)
    llm = YandexChatOpenAI(
        model=f"gpt://{settings.yandex_folder_id}/{settings.yandex_llm_model}",
        api_key=settings.yandex_api_key,
        base_url=settings.yandex_base_url,
        temperature=0.0,
        timeout=30,
        max_retries=2,
    )
    base_judge = judge_answers(llm, QUESTIONS, base_answers)
    ft_judge = judge_answers(llm, QUESTIONS, ft_answers)

    metrics = {
        "base_hello": _hello_share(base_answers),
        "ft_hello": _hello_share(ft_answers),
        "base_judge": _mean(base_judge),
        "ft_judge": _mean(ft_judge),
        "judge_n": sum(1 for v in base_judge + ft_judge if v is not None),
    }
    print(f"Metrics: {metrics}", flush=True)

    examples = [
        {
            "category": category,
            "question": question,
            "base_answer": base,
            "ft_answer": ft,
            "base_judge": bj,
            "ft_judge": fj,
        }
        for (category, question), base, ft, bj, fj in zip(
            QUESTIONS, base_answers, ft_answers, base_judge, ft_judge
        )
    ]
    (reports_dir / "finetune_examples.json").write_text(
        json.dumps({"metrics": metrics, "examples": examples}, ensure_ascii=False, indent=2)
    )

    loss_entries = _parse_loss_log(reports_dir / "lora_train_log.txt")
    train_loss_match = re.search(
        r"'train_loss': '([\d.]+)'",
        (reports_dir / "lora_train_log.txt").read_text(),
    )
    conclusions = (
        f"После дообучения модель стабильно начинает ответы со «Здравствуйте» "
        f"({metrics['ft_hello']:.0%} против {metrics['base_hello']:.0%} у base) "
        f"и чаще держит вежливый корпоративный тон (judge: {metrics['ft_judge']} "
        f"против {metrics['base_judge']}). Fine-tuned ответы короче и структурированнее: "
        f"маркированные списки, контакты поддержки в конце. Base-модель чаще "
        f"рассуждает, повторяет вопрос и выдумывает детали. Loss снизился "
        f"с {loss_entries[0]['loss']} до {loss_entries[-1]['loss']}, что "
        f"подтверждает усвоение формата ответов датасета. Важно: fine-tuned "
        f"модель, как и base, выдумывает конкретные факты (телефоны, ссылки, "
        f"номера регламентов) — SFT перенял стиль, но фактологию должен "
        f"давать RAG, а не веса модели."
        if loss_entries
        else ""
    )
    config = {
        "base_model": base_model,
        "dtype": "bfloat16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_targets": "q,k,v,o,gate,up,down proj",
        "batch_size": 1,
        "grad_accum": 8,
        "epochs": 2,
        "lr": "1e-4",
        "max_len": 768,
        "seed": 42,
        "dataset": str(settings.finetune_dataset_path),
        "dataset_size": 198,
        "train_loss": train_loss_match.group(1) if train_loss_match else "?",
        "conclusions": conclusions,
    }
    (reports_dir / "finetune_report.md").write_text(
        render_report(config, loss_entries, examples, metrics)
    )
    print(f"Report: {reports_dir / 'finetune_report.md'}", flush=True)

    lf = init_langfuse(settings)
    if lf is not None:
        with langfuse_scope(
            lambda: lf.start_as_current_observation(
                name="finetune_eval",
                as_type="span",
                input={"questions": len(QUESTIONS), "base_model": base_model},
            ),
            "finetune_eval",
        ) as root:
            trace_id = getattr(root, "trace_id", None)
            if trace_id:
                for name, value in (
                    ("base_style_score", metrics["base_judge"]),
                    ("ft_style_score", metrics["ft_judge"]),
                ):
                    if value is not None:
                        with suppress_langfuse_errors(f"create_score {name}"):
                            lf.create_score(trace_id=trace_id, name=name, value=value)
            if root is not None:
                with suppress_langfuse_errors("finetune_eval.update"):
                    root.update(output=metrics)
        with suppress_langfuse_errors("flush"):
            lf.flush()
        print("Langfuse: scores отправлены", flush=True)
    else:
        print("Langfuse: недоступен, scores не отправлены", flush=True)


if __name__ == "__main__":
    main()
