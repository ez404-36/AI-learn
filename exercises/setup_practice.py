"""Подготовка рабочих копий упражнений.

Для каждой папки упражнения `exercises/NN-.../exM_<тема>/` создаёт рядом с
`template.py` файл `work.py` — вашу личную рабочую копию. Именно в `work.py`
вы пишете решение; он игнорируется git (см. `.gitignore`), поэтому ваши
правки не засоряют `git status` и не конфликтуют при `git pull`.

Использование (из директории exercises/):

    python setup_practice.py            # создать work.py там, где их ещё нет
    python setup_practice.py --force    # перезаписать существующие work.py
    python setup_practice.py --list     # только показать, что будет создано

work.py НЕ перезаписывается без --force, чтобы не потерять ваш прогресс.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_templates() -> list[Path]:
    """Все template.py внутри папок упражнений exercises/NN-.../exM_*/."""
    return sorted(ROOT.glob("[0-9][0-9]-*/ex*/template.py"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Создать work.py из template.py")
    parser.add_argument(
        "--force", action="store_true", help="перезаписать существующие work.py"
    )
    parser.add_argument(
        "--list", action="store_true", help="показать список без создания файлов"
    )
    args = parser.parse_args()

    templates = find_templates()
    if not templates:
        print("Не найдено ни одного template.py — проверьте структуру папок.")
        return 1

    created, skipped = 0, 0
    for template in templates:
        work = template.with_name("work.py")
        rel = work.relative_to(ROOT)
        if args.list:
            print(f"  {rel}")
            continue
        if work.exists() and not args.force:
            skipped += 1
            continue
        work.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        created += 1
        print(f"  создан {rel}")

    if args.list:
        print(f"\nВсего упражнений: {len(templates)}")
        return 0

    print(
        f"\nГотово: создано {created}, пропущено (уже есть) {skipped}. "
        "Пишите решение в work.py; сверяйтесь с solution.py."
    )
    if skipped and not args.force:
        print("Чтобы пересоздать заготовки заново, запустите с --force.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
