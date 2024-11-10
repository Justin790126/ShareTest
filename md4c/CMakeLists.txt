cmake_minimum_required(VERSION 3.5)
project(MD4C)

set(MD4C_SRC md4c.cpp)
set(MD4C_HTML md4c-html.cpp entity.cpp )

add_executable(md2html main.cpp md2html.cpp cmdline.cpp  ${MD4C_SRC} ${MD4C_HTML})
