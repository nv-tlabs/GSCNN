"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import glob
import os

class ImagePage(object):
    '''
    This creates an HTML page of embedded images, useful for showing evaluation results.

    Usage:
    ip = ImagePage(html_fn)

    # Add a table with N images ...
    ip.add_table((img, descr), (img, descr), ...)

    # Generate html page
    ip.write_page()
    '''
    def __init__(self, experiment_name, html_filename):
        self.experiment_name = experiment_name
        self.html_filename = html_filename
        self.outfile = open(self.html_filename, 'w')
        self.items = []

    def _print_header(self):
        header = '''<!DOCTYPE html>
<html>
  <head>
    <title>Experiment = {}</title>
  </head>
  <body>'''.format(self.experiment_name)
        self.outfile.write(header)

    def _print_footer(self):
        self.outfile.write('''  </body>
</html>''')

    def _print_table_header(self, table_name):
        table_hdr = '''    <h3>{}</h3>
    <table border="1" style="table-layout: fixed;">
      <tr>'''.format(table_name)
        self.outfile.write(table_hdr)

    def _print_table_footer(self):
        table_ftr = '''      </tr>
    </table>'''
        self.outfile.write(table_ftr)

    def _print_table_guts(self, img_fn, descr):
        table = '''        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="{img_fn}">
              <img src="{img_fn}" style="width:768px">
            </a><br>
            <p>{descr}</p>
          </p>
        </td>'''.format(img_fn=img_fn, descr=descr)
        self.outfile.write(table)

    def add_table(self, img_label_pairs):
        self.items.append(img_label_pairs)

    def _write_table(self, table):
        img, _descr = table[0]
        self._print_table_header(os.path.basename(img))
        for img, descr in table:
            self._print_table_guts(img, descr)
        self._print_table_footer()

    def write_page(self):
        self._print_header()

        for table in self.items:
            self._write_table(table)

        self._print_footer()


def main():
    images = glob.glob('dump_imgs_train/*.png')
    images = [i for i in images if 'mask' not in i]

    ip = ImagePage('test page', 'dd.html')
    for img in images:
        basename = os.path.splitext(img)[0]
        mask_img = basename + '_mask.png'
        ip.add_table(((img, 'image'), (mask_img, 'mask')))
    ip.write_page()
