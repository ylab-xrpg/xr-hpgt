#!/bin/bash

# sudo apt install clang-format
find include/ -regex '.*\.\(cc\|cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
find src/ -regex '.*\.\(cc\|cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
find app/ -regex '.*\.\(cc\|cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
