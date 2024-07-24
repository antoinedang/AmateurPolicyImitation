#!/bin/bash
set -e

./randomly_initialize_policies.sh
./teach_policies.sh
./train_policies.sh