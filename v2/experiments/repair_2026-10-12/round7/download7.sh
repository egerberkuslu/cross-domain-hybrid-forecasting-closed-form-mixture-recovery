#!/bin/bash
# Round 7 data download (DATA7.md a1): Zenodo 10.5281/zenodo.13382427 (CC BY 4.0). Resumable (curl -C -), md5 verified against the Zenodo record API.
set -u
D=/home/ege/Desktop/zaman-serisi-yayin/data/raw/cesnet_ts24_extra
cd "$D" || exit 1
B=https://zenodo.org/api/records/13382427/files
while read -r f md5; do
  for try in 1 2 3 4 5; do
    nice -n 19 curl -sS -L -C - --retry 3 -o "$f" "$B/$f/content" || true
    got=$(md5sum "$f" | cut -d' ' -f1)
    [ "$got" = "$md5" ] && { echo "OK $f $md5"; break; }
    echo "md5 mismatch $f got $got want $md5 (try $try)"; sleep 10
  done
done <<LIST
ids_relationship.csv f034e4e7f1844e36fa7d3a4a0cbfc600
institution_subnets.tar.gz 735b5ff436b6c67556cb67e2888b14ea
ip_addresses_sample.tar.gz 08451dab2d1eddb29a79467b03289232
LIST
sha256sum ids_relationship.csv institution_subnets.tar.gz ip_addresses_sample.tar.gz > SHA256SUMS.txt
echo done
