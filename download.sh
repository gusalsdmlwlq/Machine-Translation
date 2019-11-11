#!/bin/bash

mkdir data
mkdir runs
mkdir save
wget https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz
tar -xvf de-en.tgz
mv de-en data
