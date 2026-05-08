import sys
import xml.etree.ElementTree as ET
from io import StringIO

import requests


def fetch_webpbn(num):
    data = {
        "go": 1,
        "id": num,
        "xml_clue": "on",
        "fmt": "xml",
        "xml_soln": "on",
    }
    url = "https://webpbn.com/export.cgi/webpbn%06i.sgriddler" % num
    try:
        text = requests.post(url, data, timeout=30).text
        return parse_clues(text)
    except Exception:
        return None


def parse_clues(xml_data):
    try:
        root = ET.parse(StringIO(xml_data)).getroot()
    except ET.ParseError:
        return None

    for color in root.findall(".//color"):
        if color.get('name') not in {'black', 'white'}:
            return None

    rows_formatted = ""
    columns_formatted = ""

    for clue in root.findall(".//clues"):
        formatted_clue = ""
        for line in clue.findall('./line'):
            counts = [count.text for count in line.findall('./count')]
            if counts:
                formatted_clue += ' '.join(counts) + '\n'
            else:
                formatted_clue += '0\n'

        if clue.get('type') == 'columns':
            columns_formatted += formatted_clue
        elif clue.get('type') == 'rows':
            rows_formatted += formatted_clue

    if not rows_formatted or not columns_formatted:
        return None

    return rows_formatted.strip() + "\n---\n" + columns_formatted.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fetch a single black-and-white puzzle from webpbn.com')
    parser.add_argument('id', type=int, help='webpbn puzzle ID')
    parser.add_argument('-o', '--out', help='write to this file instead of stdout')
    args = parser.parse_args()

    clue_text = fetch_webpbn(args.id)
    if clue_text is None:
        print(f"#{args.id}: not found or not black-and-white", file=sys.stderr)
        sys.exit(1)

    if args.out:
        with open(args.out, 'w') as f:
            f.write(clue_text)
    else:
        print(clue_text)


if __name__ == "__main__":
    main()
