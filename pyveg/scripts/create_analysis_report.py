from mdutils.mdutils import MdUtils
import os
import argparse
import glob
from pathlib import Path
import pypandoc


def create_markdown_pdf_report(path, collection_name):

    # getting the right suffix from the satelite to analyse
    if collection_name == 'COPERNICUS/S2':
        collection = 'Sentinel2'
        satellite_suffix = 'S2'

    elif collection_name == 'LANDSAT8':
        collection = 'Landsat8'
        satellite_suffix = 'L8'

    elif collection_name == 'LANDSAT7':
        collection = 'Landsat7'
        satellite_suffix = 'L7'

    elif collection_name == 'LANDSAT5':
        collection = 'Landsat5'
        satellite_suffix = 'L5'

    elif collection_name == 'LANDSAT4':
        collection = 'Landsat4'
        satellite_suffix = 'L4'

    else:
        raise RuntimeError("Unknown collection_name {}".format(collection_name))

    current_path = Path(path)
    current_dirs_parent = str(current_path.parent)

    try:
        coords =  str(current_path.name).split("_")
        output_path = os.path.join(path,'analysis_report_'+coords[1]+'_'+coords[2]+'_'+collection)
        mdFile = MdUtils(file_name=output_path,
                         title='Results for ' + collection + ' and coordinates: ' + coords[1] + ' (longitude) and ' +
                               coords[2] + ' (latitude)')

    except:
        # in case the directory does not have the right name with coordinates
        output_path = os.path.join(path,'analysis_report_'+collection)
        mdFile = MdUtils(file_name=output_path,
                         title='Results for ' + collection)


    # Time series figures
    mdFile.new_header(level=1, title='Time series')
    ts_path = os.path.join(path,'analysis','time-series')
    count = 0
    count = count + 1
    mdFile.new_line(mdFile.new_inline_image(text='Time series Offset50', path=os.path.join(ts_path,satellite_suffix+'-time-series_smooth.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series Offset50')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='Time series NDVI', path=os.path.join(ts_path,satellite_suffix+'-ndvi-time-series.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series NDVI')
    mdFile.new_line('')

    # STL figures
    mdFile.new_header(level=1, title='STL decomposition')
    stl_path = os.path.join(path, 'analysis', 'detrended','STL')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='STL Offset50', path=os.path.join(stl_path, satellite_suffix+'_offset50_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL Offset50')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='STL NDVI', path=os.path.join(stl_path, satellite_suffix+'_ndvi_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL NDVI')
    mdFile.new_line('')

    # Early warning figures
    mdFile.new_header(level=1, title='Early warnings analysis')
    ews_path = os.path.join(path, 'analysis', 'resiliance','deseasonalised','ewstools')

    count = count +1
    mdFile.new_line(mdFile.new_inline_image(text='EWS Offset50', path=os.path.join(ews_path, satellite_suffix+'-offset50-mean-ews.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'EWS Offset50')
    mdFile.new_line()


    count = count +1
    mdFile.new_line(
    mdFile.new_inline_image(text='EWS NDVI', path=os.path.join(ews_path, satellite_suffix+'-ndvi-mean-ews.png')))
    mdFile.new_line('')

    mdFile.new_header(level=1, title='RGB images through time')  # style is set 'atx' format by default.

    # find the RGB images, sort them and print them
    rgb_path = current_dirs_parent+'/gee_*_'+collection+'/*/PROCESSED/*RGB.png'

    count_rgb = 0
    for count_rgb, rgb_figure in enumerate(sorted(glob.glob(rgb_path)), start=1):

        rgb_figure_name = str(Path(rgb_figure).name)
        mdFile.new_line(mdFile.new_inline_image(text=rgb_figure, path=rgb_figure))
        mdFile.new_line('Figure '+str(count+count_rgb)+': '+rgb_figure_name)
        mdFile.new_line('')

    if count_rgb == 0:
        mdFile.new_line("No RGB figures available in the given path.")
        mdFile.new_line('')



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
    parser.add_argument("--collection", help="Satellite to be used in the report ",default='COPERNICUS/S2')

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
