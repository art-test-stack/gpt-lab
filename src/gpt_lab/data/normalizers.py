import regex as re

def clean_code_parrot_hook(ds):
    def clean_code_parrot(example):
        text = example.get("encoded_content") or example.get("content") or ""
        # Add any cleaning steps here if necessary

        example["text"] = text
        return example

    return ds.map(clean_code_parrot)

def clean_codeparrot_example(text):
    text = clean_loop(text)
    return text

def clean_hashtag(text):
    if text.startswith("#"):
        text = clean_hashtag(text[1:])
    return text

def clean_copyright(text):
    text = re.sub(r'Copyright \d{4}(-\d{4})? .+', '', text)
    text = re.sub(r'© \d{4} .+', '', text)
    return text

_LICENSE_REGEX = re.compile(
    r"(?is)(licensed under .*?license.*?\n)",
)

def clean_license(text):
    # license_patterns = [
    #     r'Licensed under the Apache License, Version 2.0 \(the "License"\); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0',
    #     r'This file is licensed under the MIT License.',
    #     r'This code is licensed under the GNU General Public License v3.0.',
    # ]
    # for pattern in license_patterns:
    #     text = re.sub(pattern, '', text)
    # return text
    text = _LICENSE_REGEX.sub("", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()

def clean_encoding_header(text):
    encoding_headers = [
        r'# coding:',
        r'# coding=',
        r'# -*- coding:',
        r'#!/usr/bin/env python3',
        r'# encoding:'
    ]
    for header in encoding_headers:
        if text.startswith(header):
            first_line_end = text.find('\n')
            if first_line_end != -1:
                text = text[first_line_end + 1:]
                return text
    return text

def clean_loop(text):
    previous_text = None
    while text != previous_text:
        if text == "":
            break
        previous_text = text
        text = clean_encoding_header(text)
        text = clean_copyright(text)
        text = clean_license(text)
        text = clean_hashtag(text)
    return text