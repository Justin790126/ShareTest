/*
 * MD4C: Markdown parser for C
 * (http://github.com/mity/md4c)
 *
 * Copyright (c) 2016-2024 Martin Mitáš
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "md2html.h"


/*********************************
 ***  Simple grow-able buffer  ***
 *********************************/

/* We render to a memory buffer instead of directly outputting the rendered
 * documents, as this allows using this utility for evaluating performance
 * of MD4C (--stat option). This allows us to measure just time of the parser,
 * without the I/O.
 */

/**********************
 ***  Main program  ***
 **********************/



int
main(int argc, char** argv)
{
    // FILE* in = stdin;
    // FILE* out = stdout;
    
    if(initMdParser(argc,argv) != 0) {
        exit(1);
    }

    // if(input_path != NULL && strcmp(input_path, "-") != 0) {
    //     in = fopen(input_path, "rb");
    //     if(in == NULL) {
    //         fprintf(stderr, "Cannot open %s.\n", input_path);
    //         exit(1);
    //     }
    // }
    // if(output_path != NULL && strcmp(output_path, "-") != 0) {
    //     out = fopen(output_path, "wt");
    //     if(out == NULL) {
    //         fprintf(stderr, "Cannot open %s.\n", output_path);
    //         exit(1);
    //     }
    // }

    char* html = process_string("# title");

    // char* html = process_file((input_path != NULL) ? input_path : "<stdin>", in, out);
    // ret = html ? 0 : -1;
    // if(in != stdin)
    //     fclose(in);
    // if(out != stdout)
    //     fclose(out);

    printf("------html:------\n%s\n", html);
    if (html) free(html);
    return 0;
}
