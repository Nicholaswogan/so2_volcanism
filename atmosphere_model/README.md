# For Trent

## L98-59 b example

Set up a new conda environment.

```sh
conda env create -f environment.yaml
conda activate outgassing
```

Run the example:

```sh
python input_files.py
python main.py
```

## Using Codex/Claude

If you use a coding tool like Codex or Claude, then download all the Photochem source with the commands bellow, and tell the bot to read the code when trying to make changes.

```sh
mkdir -p codex_reference && cd codex_reference
for u in \
  https://github.com/Nicholaswogan/photochem/archive/refs/tags/v0.8.2.zip \
  https://github.com/Nicholaswogan/clima/archive/refs/tags/v0.7.4.zip \
  https://github.com/Nicholaswogan/Equilibrate/archive/refs/tags/v0.2.1.zip
do
  f="$(basename "$u")"
  wget -O "$f" "$u"
  unzip -q "$f"
  rm -f "$f"
done
cd ..
```
