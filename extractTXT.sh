#!/bin/bash
FILES=/Documents/*
for f in *.pdf
do
  echo "Processing $f..."
  pdf2txt.py -o "$f".txt "$f"
done
