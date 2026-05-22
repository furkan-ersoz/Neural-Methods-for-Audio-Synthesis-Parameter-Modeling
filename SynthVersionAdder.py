#!/usr/bin/env python3
"""
SynthVersionAdder.py
Kullanım:
    python SynthVersionAdder.py           # otomatik versiyon tespiti
    python SynthVersionAdder.py 3         # SynthV3 oluşturur
    python SynthVersionAdder.py --dry-run # neyin değişeceğini gösterir
    python SynthVersionAdder.py 3 --dry-run
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SYNTHS_DIR   = os.path.join(SCRIPT_DIR, "synths")
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "Templates", "SynthTemplate")
ROOT_CMAKE   = os.path.join(SCRIPT_DIR, "CMakeLists.txt")

# Template'de bilinen bug: PARAM_NAMES 5 elemanlı tanımlanmış ama 6 isim yazılmış.
# Kopyalamadan önce düzeltilir.
PARAM_NAMES_BUG = (
    'static constexpr std::array<std::string_view, NUM_PARAMS> PARAM_NAMES = {{\n'
    '            "frequency", "amplitude", "attack", "decay", "sustain", "release"\n'
    '        }};'
)
PARAM_NAMES_FIX = (
    'static constexpr std::array<std::string_view, NUM_PARAMS> PARAM_NAMES = {{\n'
    '            "amplitude", "attack", "decay", "sustain", "release"\n'
    '        }};'
)

# PluginEditor.cpp'de resized() bug: getWidth() / 6 → / 5
RESIZE_BUG = "const int w = getWidth() / 6;"
RESIZE_FIX = "const int w = getWidth() / 5;"

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
    """Versiyon boşluklarını tespit eder. SynthV2 oluşturulacaksa SynthV1 var mı?"""
    warnings = []
    for v in range(1, version):
        path = os.path.join(SYNTHS_DIR, f"SynthV{v}")
        if not os.path.exists(path):
            warnings.append(f"Uyarı: SynthV{v} mevcut değil — boşluk oluşacak.")
    return warnings


def generate_plugin_code(version: int) -> str:
    """
    Her versiyon için deterministik ama unique 4-char plugin code üretir.
    Format: S + V + iki basamaklı versiyon (örn. SV01, SV02, ..., SV99)
    Versiyon > 99 ise rastgele harf+rakam karışımı üretilir.
    """
    if version <= 99:
        return f"SV{version:02d}"
    # Fallback: rastgele 4 karakter (büyük harf + rakam)
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


def apply_fixes_to_content(content: str, filename: str) -> tuple[str, list[str]]:
    """
    Template'deki bilinen bug'ları dosya içeriğine uygular.
    Değişiklik log'u döner.
    """
    changes = []

    # Bug 1: PARAM_NAMES aşırı eleman
    if PARAM_NAMES_BUG in content:
        content = content.replace(PARAM_NAMES_BUG, PARAM_NAMES_FIX)
        changes.append(f"  [FIX] {filename}: PARAM_NAMES 6→5 eleman düzeltildi")

    # Bug 2: resized() genişlik hesabı
    if RESIZE_BUG in content:
        content = content.replace(RESIZE_BUG, RESIZE_FIX)
        changes.append(f"  [FIX] {filename}: resized() getWidth()/6 → /5 düzeltildi")

    return content, changes


def rename_in_tree(dst_dir: str, v_str: str, dry_run: bool) -> list[str]:
    """
    Dosya adlarını ve dizin adlarını Template → V{N} olarak yeniden adlandırır.
    Doğru sıra: önce dosyalar (en derinden), sonra dizinler (en derinden yukarıya).
    """
    log = []

    # 1. Dosyaları rename et (topdown=False: en derin önce)
    for root, dirs, files in os.walk(dst_dir, topdown=False):
        for filename in files:
            if "Template" in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace("Template", v_str)
                new_path = os.path.join(root, new_filename)
                log.append(f"  [RENAME] {old_path} → {new_path}")
                if not dry_run:
                    os.rename(old_path, new_path)

    # 2. Dizinleri rename et (topdown=False: en derin önce, path güvenli)
    # os.walk tamamlandıktan sonra tüm dizin listesini topla, sırala
    all_dirs = []
    for root, dirs, _ in os.walk(dst_dir, topdown=False):
        for d in dirs:
            if "Template" in d:
                all_dirs.append(os.path.join(root, d))

    # En uzun path önce (en derin) — üst dizin rename edilmeden önce alt dizin
    all_dirs.sort(key=lambda p: -len(p))
    for old_path in all_dirs:
        # Path hâlâ geçerli mi? (üst dizin rename edilmiş olabilir)
        if not os.path.exists(old_path):
            # Güncellenmiş path'i bul
            parent = os.path.dirname(old_path)
            basename = os.path.basename(old_path)
            new_parent = parent.replace("Template", v_str)
            old_path = os.path.join(new_parent, basename)
            if not os.path.exists(old_path):
                continue
        new_path = old_path.replace("Template", v_str)
        log.append(f"  [RENAME] {old_path} → {new_path}")
        if not dry_run:
            os.rename(old_path, new_path)

    return log


def replace_contents_in_tree(dst_dir: str, v_str: str, plugin_code: str,
                              dry_run: bool) -> list[str]:
    """
    Dosya içeriklerinde Template → V{N} ve PLUGIN_CODE güncellenir.
    Bilinen bug'lar da bu aşamada düzeltilir.
    """
    log = []
    old_plugin_code_pattern = re.compile(r'(PLUGIN_CODE\s+)Stp\d*')
    old_plugin_name_in_cmake = re.compile(r'(PRODUCT_NAME\s+"SynthTemplate")')

    for root, _, files in os.walk(dst_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            content = read_file(filepath)
            if content is None:
                continue  # binary dosya, atla

            original = content
            changed = []

            # Template → V{N}
            if "Template" in content:
                content = content.replace("Template", v_str)
                changed.append("Template→" + v_str)

            # PLUGIN_CODE güncelle (CMakeLists.txt'de)
            content, code_sub_count = old_plugin_code_pattern.subn(
                rf'\g<1>{plugin_code}', content
            )
            if code_sub_count:
                changed.append(f"PLUGIN_CODE→{plugin_code}")

            # Bug fix'leri uygula (içerik dönüşümünden sonra, V{N} ile eşleşebilir)
            # Not: Bug fix pattern'ları Template→V dönüşümünden sonra güncellenir
            fixed_param_bug = PARAM_NAMES_BUG.replace("Template", v_str)
            fixed_param_fix = PARAM_NAMES_FIX.replace("Template", v_str)
            if fixed_param_bug in content:
                content = content.replace(fixed_param_bug, fixed_param_fix)
                changed.append("PARAM_NAMES 6→5 eleman")

            if RESIZE_BUG in content:
                content = content.replace(RESIZE_BUG, RESIZE_FIX)
                changed.append("resized() /6→/5")

            if content != original:
                log.append(f"  [UPDATE] {filepath}: {', '.join(changed)}")
                if not dry_run:
                    write_file(filepath, content)

    return log


def create_wrapper_cmake(dst_dir: str, dry_run: bool) -> list[str]:
    """
    synths/SynthVN/CMakeLists.txt oluşturur.
    CMake bu dosya olmadan add_subdirectory(synths/SynthVN) çağrısını işleyemez.
    İçerik: AudioEngine ve Plugin alt dizinlerini sırayla ekler.
    """
    wrapper_path = os.path.join(dst_dir, "CMakeLists.txt")
    content = (
        "# Auto-generated by SynthVersionAdder.py\n"
        "add_subdirectory(AudioEngine)\n"
        "add_subdirectory(Plugin)\n"
    )
    log = [f"  [CREATE] {wrapper_path}"]
    if not dry_run:
        write_file(wrapper_path, content)
    return log


def update_root_cmake(version: int, dry_run: bool) -> list[str]:
    """
    SS/CMakeLists.txt'e add_subdirectory(synths/SynthV{N}) ekler.
    Doğru yer: JUCE veya diğer add_subdirectory satırlarının altı,
    juce_add_plugin çağrılarının üstü.
    Zaten varsa ekleme.
    """
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

    # Ekleme noktası: son add_subdirectory satırının hemen altı
    lines = content.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("add_subdirectory"):
            insert_idx = i + 1  # bu satırın altına ekle

    if insert_idx == 0:
        # add_subdirectory hiç yoksa dosyanın sonuna ekle
        insert_idx = len(lines)

    lines.insert(insert_idx, new_line + "\n")
    new_content = "".join(lines)

    log.append(f"  [UPDATE] {ROOT_CMAKE}: '{new_line}' eklendi (satır {insert_idx + 1})")
    if not dry_run:
        write_file(ROOT_CMAKE, new_content)

    return log


def rollback(dst_dir: str) -> None:
    """Hata durumunda yarı oluşmuş dizini temizler."""
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        print(f"  [ROLLBACK] {dst_dir} silindi.")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        try:
            version = int(args[0])
            if version < 1:
                raise ValueError
        except ValueError:
            print(f"Hata: '{args[0]}' geçerli bir versiyon numarası değil (≥1 olmalı).")
            sys.exit(1)
    else:
        version = get_next_version()

    synth_name  = f"SynthV{version}"
    v_str       = f"V{version}"
    dst_dir     = os.path.join(SYNTHS_DIR, synth_name)
    plugin_code = generate_plugin_code(version)

    print(f"{'[DRY-RUN] ' if dry_run else ''}Oluşturuluyor: {synth_name}  |  PLUGIN_CODE: {plugin_code}")
    print()

    # Versiyon boşluğu uyarıları
    for w in check_version_gap(version):
        print(w)

    # Var mı kontrolü
    if os.path.exists(dst_dir):
        print(f"Hata: '{synth_name}' zaten mevcut: {dst_dir}")
        sys.exit(1)

    # Template var mı?
    if not os.path.exists(TEMPLATE_DIR):
        print(f"Hata: Template bulunamadı: {TEMPLATE_DIR}")
        sys.exit(1)

    # synths/ dizini yoksa oluştur
    if not dry_run and not os.path.exists(SYNTHS_DIR):
        os.makedirs(SYNTHS_DIR)

    all_log = []

    try:
        # 1. Kopyala
        print(f"[1/4] Template kopyalanıyor → {dst_dir}")
        if not dry_run:
            shutil.copytree(TEMPLATE_DIR, dst_dir)

        # 2. Dosya/dizin adlarını rename et
        print("[2/4] Dosya ve dizin adları yeniden adlandırılıyor...")
        log = rename_in_tree(dst_dir if not dry_run else TEMPLATE_DIR, v_str, dry_run)
        all_log += log
        for line in log:
            print(line)

        # 3. Dosya içeriklerini güncelle + bug fix
        print("[3/4] Dosya içerikleri güncelleniyor + bilinen bug'lar düzeltiliyor...")
        log = replace_contents_in_tree(dst_dir if not dry_run else TEMPLATE_DIR,
                                       v_str, plugin_code, dry_run)
        all_log += log
        for line in log:
            print(line)

        # 4. Wrapper CMakeLists.txt oluştur (synths/SynthVN/CMakeLists.txt)
        print("[4/5] Wrapper CMakeLists.txt oluşturuluyor...")
        log = create_wrapper_cmake(dst_dir if not dry_run else os.path.join(SYNTHS_DIR, synth_name), dry_run)
        all_log += log
        for line in log:
            print(line)

        # 5. Kök CMakeLists.txt güncelle
        print("[5/5] Kök CMakeLists.txt güncelleniyor...")
        log = update_root_cmake(version, dry_run)
        all_log += log
        for line in log:
            print(line)

    except Exception as e:
        print(f"\nHata oluştu: {e}")
        if not dry_run:
            rollback(dst_dir)
        sys.exit(1)

    # Özet
    print()
    if dry_run:
        print(f"[DRY-RUN TAMAMLANDI] Gerçek değişiklik yapılmadı. {len(all_log)} işlem simüle edildi.")
    else:
        print(f"Tamamlandı: {dst_dir}")
        print()
        # Oluşturulan ağacı göster
        for root, dirs, files in os.walk(dst_dir):
            level = root.replace(dst_dir, "").count(os.sep)
            indent = "    " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files):
                print(f"{indent}    {f}")


if __name__ == "__main__":
    main()