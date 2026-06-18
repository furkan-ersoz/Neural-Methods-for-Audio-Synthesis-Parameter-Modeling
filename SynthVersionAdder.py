#!/usr/bin/env python3
"""
SynthVersionAdder.py
Kullanım:
    python SynthVersionAdder.py           # otomatik versiyon tespiti, Template'den kopyalar
    python SynthVersionAdder.py 2         # SynthV2'den kopyalayarak sonraki versiyonu oluşturur
    python SynthVersionAdder.py --dry-run # neyin değişeceğini gösterir
    python SynthVersionAdder.py 2 --dry-run
"""

import os
import re
import sys
import shutil
import string
import random

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SYNTHS_DIR    = os.path.join(SCRIPT_DIR, "synths")
TEMPLATE_DIR  = os.path.join(SCRIPT_DIR, "Templates", "SynthTemplate")
ROOT_CMAKE    = os.path.join(SCRIPT_DIR, "CMakeLists.txt")
DATAGEN_CMAKE = os.path.join(SCRIPT_DIR, "DataGenerator", "CMakeLists.txt")

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def get_next_version() -> int:
    """synths/ içindeki mevcut SynthVN dizinlerine bakarak sıradaki versiyonu döner."""
    if not os.path.exists(SYNTHS_DIR):
        return 1
    existing = [
        d for d in os.listdir(SYNTHS_DIR)
        if re.match(r"SynthV\d+$", d) and os.path.isdir(os.path.join(SYNTHS_DIR, d))
    ]
    if not existing:
        return 1
    numbers = [int(re.search(r"\d+", d).group()) for d in existing]
    return max(numbers) + 1


def check_version_gap(version: int) -> list[str]:
    """Versiyon boşluklarını tespit eder."""
    warnings = []
    for v in range(1, version):
        path = os.path.join(SYNTHS_DIR, f"SynthV{v}")
        if not os.path.exists(path):
            warnings.append(f"Uyarı: SynthV{v} mevcut değil — boşluk oluşacak.")
    return warnings


def generate_plugin_code(version: int) -> str:
    """
    Her versiyon için deterministik ama unique 4-char plugin code üretir.
    Format: SV + iki basamaklı versiyon (örn. SV01, SV02, ..., SV99)
    Versiyon > 99 ise rastgele harf+rakam karışımı üretilir.
    """
    if version <= 99:
        return f"SV{version:02d}"
    chars = string.ascii_uppercase + string.digits
    random.seed(version)
    return "".join(random.choices(chars, k=4))


def read_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, IsADirectoryError, PermissionError):
        return None


def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def rename_in_tree(dst_dir: str, src_str: str, v_str: str, dry_run: bool) -> list[str]:
    """
    Dosya adlarını ve dizin adlarını src_str → V{N} olarak yeniden adlandırır.
    Doğru sıra: önce dosyalar (en derinden), sonra dizinler (en derinden yukarıya).
    """
    log = []

    # 1. Dosyaları rename et (topdown=False: en derin önce)
    for root, dirs, files in os.walk(dst_dir, topdown=False):
        for filename in files:
            if filename.startswith("._"):  # macOS resource fork — atla
                continue
            if src_str in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(src_str, v_str)
                new_path = os.path.join(root, new_filename)
                log.append(f"  [RENAME] {old_path} → {new_path}")
                if not dry_run:
                    os.rename(old_path, new_path)

    # 2. Dizinleri rename et (en derin önce)
    all_dirs = []
    for root, dirs, _ in os.walk(dst_dir, topdown=False):
        for d in dirs:
            if src_str in d:
                all_dirs.append(os.path.join(root, d))

    all_dirs.sort(key=lambda p: -len(p))
    for old_path in all_dirs:
        if not os.path.exists(old_path):
            parent = os.path.dirname(old_path)
            basename = os.path.basename(old_path)
            new_parent = parent.replace(src_str, v_str)
            old_path = os.path.join(new_parent, basename)
            if not os.path.exists(old_path):
                continue
        new_path = old_path.replace(src_str, v_str)
        log.append(f"  [RENAME] {old_path} → {new_path}")
        if not dry_run:
            os.rename(old_path, new_path)

    return log


def replace_contents_in_tree(dst_dir: str, src_str: str, v_str: str, version: int,
                              plugin_code: str, dry_run: bool) -> list[str]:
    """
    Dosya içeriklerinde Template → V{N} ve PLUGIN_CODE güncellenir.
    SynthEngineVN.cpp'deki ##SYNTH_REGISTRATION## token'ı gerçek kayıt bloğuyla değiştirilir.
    """
    log = []
    old_plugin_code_pattern = re.compile(r'(PLUGIN_CODE\s+)Stp\d*')

    synth_name = f"SynthV{version}"
    namespace  = f"SynthV{version}"
    registration_block = (
        f"// ── SynthRegistry registration ────────────────────────────────────────────────\n"
        f"namespace {{\n"
        f"    const bool _registered = []() {{\n"
        f'        SynthRegistry::instance().registerSynth("{synth_name}", []() {{\n'
        f"            return std::make_unique<{namespace}::SynthEngine>();\n"
        f"        }});\n"
        f"        return true;\n"
        f"    }}();\n"
        f"}}"
    )

    for root, _, files in os.walk(dst_dir):
        for filename in files:
            if filename.startswith("._"):  # macOS resource fork — atla
                continue
            filepath = os.path.join(root, filename)
            content = read_file(filepath)
            if content is None:
                continue

            original = content
            changed = []

            # src_str → V{N}
            if src_str in content:
                content = content.replace(src_str, v_str)
                changed.append(src_str + "→" + v_str)

            # PLUGIN_CODE güncelle (CMakeLists.txt'de)
            content, code_sub_count = old_plugin_code_pattern.subn(
                rf'\g<1>{plugin_code}', content
            )
            if code_sub_count:
                changed.append(f"PLUGIN_CODE→{plugin_code}")

            # ##SYNTH_REGISTRATION## token'ını gerçek kayıt bloğuyla değiştir
            if "##SYNTH_REGISTRATION##" in content:
                content = content.replace("// ##SYNTH_REGISTRATION##", registration_block)
                # Sonraki açıklama satırını da kaldır
                content = content.replace(
                    "\n// SynthVersionAdder.py replaces the token above with the registration block.",
                    ""
                )
                changed.append("SynthRegistry kaydı eklendi")

            if content != original:
                log.append(f"  [UPDATE] {filepath}: {', '.join(changed)}")
                if not dry_run:
                    write_file(filepath, content)

    return log


def create_wrapper_cmake(dst_dir: str, dry_run: bool) -> list[str]:
    """synths/SynthVN/CMakeLists.txt oluşturur (varsa üzerine yazar)."""
    wrapper_path = os.path.join(dst_dir, "CMakeLists.txt")
    content = (
        "# Auto-generated by SynthVersionAdder.py\n"
        "add_subdirectory(AudioEngine)\n"
        "add_subdirectory(Plugin)\n"
    )
    action = "OVERWRITE" if os.path.exists(wrapper_path) else "CREATE"
    log = [f"  [{action}] {wrapper_path}"]
    if not dry_run:
        os.makedirs(dst_dir, exist_ok=True)
        write_file(wrapper_path, content)
    return log


def update_root_cmake(version: int, dry_run: bool) -> list[str]:
    """SS/CMakeLists.txt'e add_subdirectory(synths/SynthV{N}) ekler."""
    log = []
    new_line = f"add_subdirectory(synths/SynthV{version})"

    if not os.path.exists(ROOT_CMAKE):
        log.append(f"  [SKIP] Kök CMakeLists.txt bulunamadı: {ROOT_CMAKE}")
        return log

    content = read_file(ROOT_CMAKE)
    if content is None:
        log.append("  [SKIP] Kök CMakeLists.txt okunamadı.")
        return log

    if new_line in content:
        log.append(f"  [SKIP] Kök CMakeLists.txt zaten '{new_line}' içeriyor.")
        return log

    lines = content.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("add_subdirectory"):
            insert_idx = i + 1

    if insert_idx == 0:
        insert_idx = len(lines)

    lines.insert(insert_idx, new_line + "\n")
    new_content = "".join(lines)

    log.append(f"  [UPDATE] {ROOT_CMAKE}: '{new_line}' eklendi (satır {insert_idx + 1})")
    if not dry_run:
        write_file(ROOT_CMAKE, new_content)

    return log


def update_datagen_cmake(version: int, dry_run: bool) -> list[str]:
    """
    DataGenerator/CMakeLists.txt'e SynthEngineV{N}Core bağlantısını ekler.
    Yeni satır, mevcut son SynthEngineV*Core satırının hemen altına eklenir.
    """
    log = []
    new_target = f"SynthEngineV{version}Core"
    new_line   = f"        {new_target}                # SynthV{version} — registered at static-init time\n"

    if not os.path.exists(DATAGEN_CMAKE):
        log.append(f"  [SKIP] DataGenerator/CMakeLists.txt bulunamadı: {DATAGEN_CMAKE}")
        return log

    content = read_file(DATAGEN_CMAKE)
    if content is None:
        log.append("  [SKIP] DataGenerator/CMakeLists.txt okunamadı.")
        return log

    if new_target in content:
        log.append(f"  [SKIP] DataGenerator/CMakeLists.txt zaten '{new_target}' içeriyor.")
        return log

    lines = content.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        if re.search(r"SynthEngineV\d+Core", line):
            insert_idx = i + 1

    if insert_idx == 0:
        # First synth — use the anchor comment as insertion point
        for i, line in enumerate(lines):
            if "SynthVersionAdder.py appends" in line:
                insert_idx = i
                break

    if insert_idx == 0:
        log.append(f"  [SKIP] Anchor bulunamadı; manuel ekleme gerekli.")
        return log

    lines.insert(insert_idx, new_line)
    new_content = "".join(lines)

    log.append(f"  [UPDATE] {DATAGEN_CMAKE}: '{new_target}' eklendi (satır {insert_idx + 1})")
    if not dry_run:
        write_file(DATAGEN_CMAKE, new_content)

    return log


def rollback(dst_dir: str) -> None:
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        print(f"  [ROLLBACK] {dst_dir} silindi.")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    src_version = None
    if args:
        try:
            src_version = int(args[0])
            if src_version < 1:
                raise ValueError
        except ValueError:
            print(f"Hata: '{args[0]}' geçerli bir versiyon numarası değil (≥1 olmalı).")
            sys.exit(1)

    version     = get_next_version()
    synth_name  = f"SynthV{version}"
    v_str       = f"V{version}"
    dst_dir     = os.path.join(SYNTHS_DIR, synth_name)
    plugin_code = generate_plugin_code(version)

    if src_version is not None:
        src_dir = os.path.join(SYNTHS_DIR, f"SynthV{src_version}")
        src_str = f"V{src_version}"
        src_label = f"SynthV{src_version}"
    else:
        src_dir = TEMPLATE_DIR
        src_str = "Template"
        src_label = "Template"

    print(f"{'[DRY-RUN] ' if dry_run else ''}Oluşturuluyor: {synth_name}  (kaynak: {src_label})  |  PLUGIN_CODE: {plugin_code}")
    print()

    for w in check_version_gap(version):
        print(w)

    if os.path.exists(dst_dir):
        print(f"Hata: '{synth_name}' zaten mevcut: {dst_dir}")
        sys.exit(1)

    if not os.path.exists(src_dir):
        print(f"Hata: Kaynak bulunamadı: {src_dir}")
        sys.exit(1)

    if not dry_run and not os.path.exists(SYNTHS_DIR):
        os.makedirs(SYNTHS_DIR)

    all_log = []

    try:
        # 1. Kopyala (datasets/ ve build/ atlanır — büyük/geçici dizinler)
        SKIP_DIRS = {"datasets", "build", ".git"}
        def _ignore(src, names):
            skip = set()
            for n in names:
                if n.startswith("._") or n in SKIP_DIRS:
                    skip.add(n)
            return skip

        print(f"[1/6] {src_label} kopyalanıyor → {dst_dir}  (datasets/ ve build/ atlanıyor)")
        if not dry_run:
            shutil.copytree(src_dir, dst_dir, ignore=_ignore)

        # 2. Dosya/dizin adlarını rename et
        print("[2/6] Dosya ve dizin adları yeniden adlandırılıyor...")
        log = rename_in_tree(dst_dir if not dry_run else src_dir, src_str, v_str, dry_run)
        all_log += log
        for line in log:
            print(line)

        # 3. Dosya içeriklerini güncelle
        print("[3/6] Dosya içerikleri güncelleniyor...")
        log = replace_contents_in_tree(dst_dir if not dry_run else src_dir,
                                       src_str, v_str, version, plugin_code, dry_run)
        all_log += log
        for line in log:
            print(line)

        # 4. Wrapper CMakeLists.txt oluştur (synths/SynthVN/CMakeLists.txt)
        print("[4/6] Wrapper CMakeLists.txt oluşturuluyor...")
        log = create_wrapper_cmake(dst_dir if not dry_run else os.path.join(SYNTHS_DIR, synth_name), dry_run)
        all_log += log
        for line in log:
            print(line)

        # 5. Kök CMakeLists.txt güncelle
        print("[5/6] Kök CMakeLists.txt güncelleniyor...")
        log = update_root_cmake(version, dry_run)
        all_log += log
        for line in log:
            print(line)

        # 6. DataGenerator CMakeLists.txt güncelle
        print("[6/6] DataGenerator/CMakeLists.txt güncelleniyor...")
        log = update_datagen_cmake(version, dry_run)
        all_log += log
        for line in log:
            print(line)

    except Exception as e:
        print(f"\nHata oluştu: {e}")
        if not dry_run:
            rollback(dst_dir)
        sys.exit(1)

    print()
    if dry_run:
        print(f"[DRY-RUN TAMAMLANDI] Gerçek değişiklik yapılmadı. {len(all_log)} işlem simüle edildi.")
    else:
        print(f"Tamamlandı: {dst_dir}")
        print()
        for root, dirs, files in os.walk(dst_dir):
            level = root.replace(dst_dir, "").count(os.sep)
            indent = "    " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files):
                print(f"{indent}    {f}")


if __name__ == "__main__":
    main()
