import pkg_resources
import subprocess

# Inisialisasi list untuk menyimpan nama paket yang akan dihapus
packages_to_remove = []

# Temukan semua paket dengan versi None
for dist in pkg_resources.working_set:
    if dist.version is None:
        print(f"Found package with None version: {dist.project_name}")
        packages_to_remove.append(dist.project_name)

# Simpan daftar paket ke dalam file untuk reinstall
with open("packages_to_reinstall.txt", "w") as f:
    for package in packages_to_remove:
        f.write(f"{package}\n")

# Uninstall semua paket dengan versi None
for package in packages_to_remove:
    subprocess.call(['pip', 'uninstall', package, '-y'])

print("\nPaket dengan versi None telah dihapus.")
print("Nama-nama paket yang dihapus telah disimpan di 'packages_to_reinstall.txt'")