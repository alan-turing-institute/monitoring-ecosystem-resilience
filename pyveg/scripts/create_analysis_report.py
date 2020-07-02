from mdutils.mdutils import MdUtils
import os
import argparse
import glob
from pathlib import Path
import pypandoc


def create_markdown_pdf_report(path, collection ='Sentinel2'):

    # getting the right suffix from the satelite to analyse
    if collection == 'Sentinel2':
        satellite = 'S2'
    elif collection == 'Landsat7':
        satellite = 'L7'

    current_path = Path(path)
    current_dirs_parent = str(current_path.parent)
    coords =  str(current_path.name).split("_")

    output_path = os.path.join(path,'analysis_report_'+coords[1]+'_'+coords[2])


    mdFile = MdUtils(file_name=output_path, title='Results for ' + collection+' and coordinates: '+coords[1]+' (longitude) and '+coords[2]+' (latitude)')
    mdFile.new_header(level=1, title='RGB images through time')  # style is set 'atx' format by default.

    # find the RGB images, sort them and print them
    rgb_path = current_dirs_parent+'/gee_*_'+collection+'/*/PROCESSED/*RGB.png'

    for count, rgb_figure in enumerate(sorted(glob.glob(rgb_path)), start=1):

        rgb_figure_name = str(Path(rgb_figure).name)
        mdFile.new_line(mdFile.new_inline_image(text=rgb_figure, path=os.path.join(rgb_path,rgb_figure)))
        mdFile.new_line('Figure '+str(count)+': '+rgb_figure_name)
        mdFile.new_line('')


    # Time series figures
    mdFile.new_header(level=1, title='Time series')
    ts_path = os.path.join(path,'analysis','time-series')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='Time series Offset50', path=os.path.join(ts_path,satellite+'-time-series_smooth.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series Offset50')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='Time series NDVI', path=os.path.join(ts_path,satellite+'-ndvi-time-series.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series NDVI')
    mdFile.new_line('')

    # STL figures
    mdFile.new_header(level=1, title='STL decomposition')
    stl_path = os.path.join(path, 'analysis', 'detrended','STL')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='STL Offset50', path=os.path.join(stl_path, satellite+'_offset50_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL Offset50')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='STL NDVI', path=os.path.join(stl_path, satellite+'_ndvi_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL NDVI')
    mdFile.new_line('')

    # Early warning figures
    mdFile.new_header(level=1, title='Early warnings analysis')
    ews_path = os.path.join(path, 'analysis', 'resiliance','deseasonalised','ewstools')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='EWS Offset50', path=os.path.join(ews_path, satellite+'-offset50-mean-ews.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'EWS Offset50')

    count = count +1
    mdFile.new_line(
    mdFile.new_inline_image(text='EWS NDVI', path=os.path.join(ews_path, satellite+'-ndvi-mean-ews.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'EWS NDVI')

    # Create a table of contents and save file
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()


    output = pypandoc.convert_file(output_path+'.md', 'pdf', outputfile=output_path+".pdf")
    assert output == ""

def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(description="Collect all figures from analysis and get them into report")
    parser.add_argument("--input_dir", help="results directory from `download_gee_data` script, containing `results_summary.json` and `analysis` directory")
    parser.add_argument("--collection", help="Satellite to be used in the report ",default='Sentinel2')

    print('-' * 35)
    print('Running create_analysis_report.py')
    print('-' * 35)

    # parse args
    args = parser.parse_args()
    input_dir = args.input_dir
    collection = args.collection

    # run markdown code
    create_markdown_pdf_report(input_dir, collection)


if __name__ == "__main__":
    main()

