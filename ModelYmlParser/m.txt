cmake_minimum_required(VERSION 3.5)
project(MD4C)

set(MD4C_SRC md4c.c md4c.h)
set(MD4C_HTML md4c-html.c md4c-html.h entity.c entity.h)

add_executable(md2html md2html.c cmdline.c cmdline.h ${MD4C_SRC} ${MD4C_HTML})
