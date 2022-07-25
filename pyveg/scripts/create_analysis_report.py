import argparse
import glob
import json
import os
import tempfile
from pathlib import Path

import pypandoc
from mdutils.mdutils import MdUtils

from pyveg.src.azure_utils import download_rgb, download_summary_json


def get_collection_and_suffix(collection_name):
    """
    Lookup collection and suffix based on the name of the collection as used by GEE

    Parameters
    ==========
    collection_name: str, GEE name of the collection, eg. 'COPERNICUS/S2'

    Returns
    =======
    collection: str, user-friendly name of the collection, e.g. 'Sentinel2'
    suffix:  str, contraction of collection name, used in the filenames of plots
    """

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
    return collection, satellite_suffix


def add_time_series_plots(mdFile, analysis_plots_location, analysis_plots_location_type, satellite_suffix):
    if analysis_plots_location_type != "local":
        print("Only local disk location for analysis plots is currently supported")
        return mdFile, 0
    fig_count = 0
    # Time series figures
    mdFile.new_header(level=1, title='Time series')
    ts_path = os.path.join(analysis_plots_location,'analysis','time-series')
    fig_count += 1
    mdFile.new_line(mdFile.new_inline_image(text='Time series Offset50', path=os.path.join(ts_path,satellite_suffix+'-time-series_smooth.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series Offset50')

    fig_count += 1
    mdFile.new_line(mdFile.new_inline_image(text='Time series NDVI', path=os.path.join(ts_path,satellite_suffix+'-ndvi-time-series.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'Time series NDVI')
    mdFile.new_line('')

    # STL figures
    mdFile.new_header(level=1, title='STL decomposition')
    stl_path = os.path.join(analysis_plots_location, 'analysis', 'detrended','STL')
    fig_count += 1

    mdFile.new_line(mdFile.new_inline_image(text='STL Offset50', path=os.path.join(stl_path, satellite_suffix+'_offset50_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL Offset50')
    fig_count += 1
    mdFile.new_line(mdFile.new_inline_image(text='STL NDVI', path=os.path.join(stl_path, satellite_suffix+'_ndvi_mean_STL_decomposition.png')))
    #mdFile.new_paragraph('Figure '+str(count)+': '+'STL NDVI')
    mdFile.new_line('')

    if os.path.exists(os.path.join(analysis_plots_location, 'analysis', 'resiliance','deseasonalised','ewstools')):
        # Early warning figures
        mdFile.new_header(level=1, title='Early warnings analysis')
        ews_path = os.path.join(analysis_plots_location, 'analysis', 'resiliance','deseasonalised','ewstools')
        fig_count += 1
        mdFile.new_line(mdFile.new_inline_image(text='EWS Offset50', path=os.path.join(ews_path, satellite_suffix+'-offset50-mean-ews.png')))
        #mdFile.new_paragraph('Figure '+str(count)+': '+'EWS Offset50')
        mdFile.new_line()

        fig_count += 1
        mdFile.new_line(
        mdFile.new_inline_image(text='EWS NDVI', path=os.path.join(ews_path, satellite_suffix+'-ndvi-mean-ews.png')))
        mdFile.new_line('')

    mdFile.new_header(level=1, title='Average annual time series CB fit')
    fig_count += 1
    mdFile.new_line(
        mdFile.new_inline_image(text='Offset50 CB fit', path=os.path.join(analysis_plots_location,'analysis', 'fit_ts_CB_S2_offset50_mean.png')))
    mdFile.new_line('')
    fig_count += 1
    mdFile.new_line(
        mdFile.new_inline_image(text='NDVI CB fit',
                                path=os.path.join(analysis_plots_location, 'analysis','fit_ts_CB_S2_ndvi_mean.png')))
    mdFile.new_line('')
    fig_count += 1
    mdFile.new_line(
        mdFile.new_inline_image(text='total precipitation CB fit',
                                path=os.path.join(analysis_plots_location, 'analysis','fit_ts_CB_total_precipitation.png')))
    mdFile.new_line('')

    return mdFile, fig_count


def add_rgb_images(mdFile, rgb_location, rgb_location_type, fig_count):
    mdFile.new_header(level=1, title='RGB images through time')  # style is set 'atx' format by default.
    rgb_filenames = []
    if rgb_location_type == "local":
        for root, dirs, files in os.walk(rgb_location):
            for filename in files:
                if filename.endswith("RGB.png") and os.path.basename(filename).startswith('sub')==False:
                    rgb_filenames.append(os.path.join(rgb_location, root, filename))
    elif rgb_location_type == "azure":
        tmpdir = tempfile.mkdtemp()
        download_rgb(rgb_location, tmpdir)
        rgb_filenames = [os.path.join(tmpdir, fname) for fname in os.listdir(tmpdir)]
    else:
        print("""
        Trying to add RGB images to report - unknown value for rgb_location_type - {}.
        Currently accepted values are ['local','azure']
        """.format(rgb_location_type))
        return mdFile
    rgb_filenames.sort()
    for i, rgb_figure in enumerate(rgb_filenames):
        rgb_figure_name = os.path.basename(rgb_figure)
        mdFile.new_line(mdFile.new_inline_image(text=rgb_figure, path=rgb_figure))
        mdFile.new_line('Figure '+str(fig_count)+': '+rgb_figure_name)
        mdFile.new_line('')
        fig_count += 1
    return mdFile


def create_markdown_pdf_report(analysis_plots_location,
                               analysis_plots_location_type,
                               rgb_location,
                               rgb_location_type,
                               do_timeseries,
                               output_dir=None,
                               collection_name=None,
                               metadata=None):
    if not metadata:
        if os.path.exists(os.path.join(analysis_plots_location, "results_summary.json")):
            try:
                metadata = json.load(open(os.path.join(analysis_plots_location, "results_summary.json")))["metadata"]
            except:
                print("Couldn't retrieve metadata from {}".format(analysis_plots_location, "results_summary.json"))
        elif rgb_location_type == "azure":
            tmpdir = tempfile.mkdtemp()
            download_summary_json(rgb_location, tmpdir)
            try:
                metadata = json.load(open(os.path.join(tmpdir, "results_summary.json")))["metadata"]
            except:
                print("Couldn't retrieve metadata from {}".format(tmpdir, "results_summary.json"))
    if metadata and not collection_name:
        collection_name = metadata['collection']
    if not collection_name:
        raise RuntimeError("please provide either a metadata dictionary or a collection name (e.g. 'COPERNICUS/S2')")
    collection, satellite_suffix = get_collection_and_suffix(collection_name)

    if metadata:
        coordinates = (metadata['longitude'], metadata['latitude'])
    if not coordinates:
        raise RuntimeError("Unable to find coordinates from path.  Please provide a metadata dict.")

    if not output_dir:
        # put the output report in the same directory as the analysis plots
        output_dir = analysis_plots_location
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir,'analysis_report_{}_{}_{}'.format(coordinates[0],coordinates[1],collection))

    # create the markdown file
    mdFile = MdUtils(file_name=output_path,
                     title='Results for {} and coordinates {} (longitude) and {} (latitude)'\
                     .format(collection,
                            coordinates[0],
                            coordinates[1]))

    fig_count = 0
    if do_timeseries:
        mdFile, fig_count = add_time_series_plots(mdFile,
                                                  analysis_plots_location,
                                                  analysis_plots_location_type,
                                                  satellite_suffix)

    # add RGB images
    mdFile = add_rgb_images(mdFile, rgb_location, rgb_location_type, fig_count)

    # Create a table of contents and save file
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    print ('Converting to pdf.')
    output = pypandoc.convert_file(output_path+'.md', 'pdf', outputfile=output_path+".pdf")
    assert output == ""


def main():
    """
    CLI interface for gee data analysis.
    """
    parser = argparse.ArgumentParser(description="Collect all figures from analysis and get them into report")
    parser.add_argument("--input_analysis_plots_location", help="results directory from `download_gee_data` script, containing `results_summary.json` and `analysis` directory")
    parser.add_argument("--input_analysis_plots_location_type", help="currently supports 'local' or 'azure'",default="local")
    parser.add_argument("--input_rgb_location", help="location of the RGB plots - either an Azure container or a local directory")
    parser.add_argument("--input_rgb_location_type", help="currently supports 'local' or 'azure'", default="local")
    parser.add_argument("--output_dir", help="(optional) directory to store output report.  If not specified, will use input_analysis_plots_location")
    parser.add_argument("--do_timeseries", action="store_true", help="include time-series plots in the report")
    parser.add_argument("--collection", help="Satellite to be used in the report ",default='COPERNICUS/S2')

    print('-' * 35)
    print('Running create_analysis_report.py')
    print('-' * 35)

    # parse args
    args = parser.parse_args()
    input_analysis_plots_location = args.input_analysis_plots_location
    input_analysis_plots_location_type = args.input_analysis_plots_location_type
    input_rgb_location = args.input_rgb_location
    input_rgb_location_type = args.input_rgb_location_type
    output_dir = args.output_dir
    do_timeseries = args.do_timeseries if args.do_timeseries else False
    collection = args.collection

    # run markdown code
    create_markdown_pdf_report(input_analysis_plots_location,
                               input_analysis_plots_location_type,
                               input_rgb_location,
                               input_rgb_location_type,
                               do_timeseries,
                               output_dir,
                               collection)


if __name__ == "__main__":
    main()
