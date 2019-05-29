import json
import csv
from tqdm import tqdm
import numpy as np


def parse_data(source1, output_path, numData):
    # setup an array for writing each row in the csv file
    rows = []
    # extract fields from business json data set #
    # setup an array for storing each json entry
    business_data = []
    # setup an array for headers we are not using strictly
    removed_header = ['item_attributes']
    # setup an array for headers we are adding
    business_header_additions = ['Sunday_Open', 'Sunday_Close', 'Monday_Open', 'Monday_Close', 'Tuesday_Open',
                                 'Tuesday_Close', 'Wednesday_Open', 'Wednesday_Close', 'Thursday_Open',
                                 'Thursday_Close', 'Friday_Open', 'Friday_Close', 'Saturday_Open', 'Saturday_Close',
                                 'NoiseLevel', 'RestaurantsAttire', "RestaurantsTakeOut", 'RestaurantsReservations',
                                 'RestaurantsDelivery', 'Alcohol','RestaurantsPriceRange2', 'BikeParking',
                                 'HappyHour', 'OutdoorSeating','RestaurantsGoodForGroups',
                                 'HasTV', 'Caters', 'GoodForKids', 'BusinessAcceptsCreditCards',
                                 'WiFi', 'GoodForDancing', 'Smoking', 'RestaurantsTableService', 'Corkage', 'CoatCheck', "BYOB",
                                 'Parking_Street', 'Parking_Valet', 'Parking_Lot', 'Parking_Garage', 'Parking_Validated']
    data = 0
    with open(source1) as f:
        # for each line in the json file
        for line in f:
            # store the line in the array for manipulation
            data = json.loads(line)
            print("count")
    # close the reader
    f.close()
    for i in range(numData):
        business_data.append(data[str(i)])
    # append the initial keys as csv headers
    header = sorted(business_data[0].keys())
    for h in removed_header:
        header.remove(h)
    orig_header = header
    # remove keys from the business data that we are not using strictly
    # for headers in business_header_removals:
    #     header.remove(headers)
    # append the additional business related csv headers

    print('processing data in the business dataset...')
    # for every entry in the business data array
    true_count = 0
    false_count = 0
    obj_count = 0
    nan_count = 0
    for entry in tqdm(range(0, len(business_data))):
        row = []
        for item in orig_header:
                if item not in removed_header:
                    row.append(business_data[entry][item])
        # set up an array for the days of the week
        days_of_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        # iterate through the days of the week to extract the open and close times
        for time in days_of_week:
            # if a time is available
            if business_data[entry]['item_hours'] is not None:
                if time in business_data[entry]['item_hours'].keys():
                    # append the open time
                    if "-" in business_data[entry]['item_hours'][time]:
                        row.append(business_data[entry]['item_hours'][time].split('-')[0])
                    # append the closing time
                        row.append(business_data[entry]['item_hours'][time].split('-')[1])

                    else:
                        row.append(business_data[entry]['item_hours'][time])
                    # append the closing time
                        row.append(business_data[entry]['item_hours'][time])
                # else if a time is not available
                else:
                    # append NA for the open time
                    row.append(np.nan)
                    # append NA for the closing time
                    row.append(np.nan)
            else:
                # append NA for the open time
                row.append(np.nan)
                # append NA for the closing time
                row.append(np.nan)
        # extract the attributes of interest
        attributes = ['NoiseLevel', 'RestaurantsAttire', "RestaurantsTakeOut", 'RestaurantsReservations',
                      'RestaurantsDelivery', 'Alcohol','RestaurantsPriceRange2', 'BikeParking',
                      'HappyHour', 'OutdoorSeating','RestaurantsGoodForGroups',
                      'HasTV', 'Caters', 'GoodForKids', 'BusinessAcceptsCreditCards',
                      'WiFi', 'GoodForDancing', 'Smoking', 'RestaurantsTableService', 'Corkage', 'CoatCheck', "BYOB"]
        # for each attribute that is not nested
        for attribute in attributes:
            # if there is an attribute
            if business_data[entry]['item_attributes'] is not None:
                if attribute in business_data[entry]['item_attributes'].keys():
                    # if the attribute contains true
                    if business_data[entry]['item_attributes'][attribute] == "True":
                        row.append(1)
                        if attribute == "Caters":
                            true_count +=1
                    # else if the attribute contains false
                    elif business_data[entry]['item_attributes'][attribute] == "False":
                        row.append(0)
                        if attribute == "Caters":
                            false_count +=1
                    # else if the attribute is non-empty and not true of false
                    elif business_data[entry]['item_attributes'][attribute] is not None:
                        row.append(str(business_data[entry]['item_attributes'][attribute]).replace('u', ''))
                # else of the attribute is not available
                    else:
                        row.append(np.nan)
                        if attribute == "Caters":
                            nan_count +=1
                else:
                    # append NA for the attribute
                    row.append(np.nan)
                    if attribute == "Caters":
                        nan_count +=1
            else:
                row.append(np.nan)
                if attribute == "Caters":
                    nan_count +=1
        # extract the parking attributes
        parking_attributes = ['street', 'valet', 'lot', 'garage', 'validated']
        # for each parking attribute
        for attribute in parking_attributes:
            # if there are parking attributes
            if business_data[entry]['item_attributes'] is not None:
                if 'Parking' in business_data[entry]['item_attributes']:
                    # if the parking attribute exists
                    if attribute in business_data[entry]['item_attributes']['Parking']:
                        # if the parking attribute is true
                        if business_data[entry]['item_attributes']['Parking'][attribute] == "True":
                            row.append(1)
                        # if the parking attribute is false
                        elif business_data[entry]['item_attributes']['Parking'][attribute] == "False":
                            row.append(0)
                        # note that the parking attributes are all true/false so no need for is not None elif
                    # else if the specific attribute is not available
                        else:
                            row.append(np.nan)
                    else:
                        row.append(np.nan)
                # else if the parking attribute is not availablestr(item).replace('\n', ' ')
                else:
                    row.append(np.nan)
            else:
                row.append(np.nan)
        # extract the music attributes
        # music_attributes = ['dj', 'karaoke', 'video', 'live', 'jukebox', 'background_music']
        # # for each music attribute
        # for attribute in music_attributes:
        #     # if there are music attributes
        #     if 'Music' in business_data[entry]['item_attributes']:
        #         # if the music attribute exists
        #         if attribute in business_data[entry]['item_attributes']['Music'].keys():
        #             # if the music attribute is true
        #             if business_data[entry]['item_attributes']['Music'][attribute] is True:
        #                 row.append(1)
        #             # if the music attribute is false
        #             elif business_data[entry]['item_attributes']['Music'][attribute] is False:
        #                 row.append(0)
        #             # note that the music attributes are all true/false so no need for is not None elif
        #         # else if the specific attribute is not available
        #         else:
        #             row.append('NA')
        #     # else if the music attribute is not available
        #     else:
        #         row.append('NA')

        # extract the categories
        # categories_of_interest = ['Restaurants', 'Sandwiches', 'Fast Food', 'Nightlife', 'Pizza', 'Bars', 'Mexican',
        #                           'Food', 'American (Traditional)', 'Burgers', 'Chinese', 'Italian',
        #                           'American (New)', 'Breakfast & Brunch', 'Thai', 'Indian', 'Sushi Bars', 'Korean',
        #                           'Mediterranean', 'Japanese', 'Seafood', 'Middle Eastern', 'Pakistani', 'Barbeque',
        #                           'Vietnamese', 'Asian Fusion', 'Diners', 'Greek', 'Vegetarian']
        # # for each category of interest
        # for category in categories_of_interest:
        #     # if the category is in the category entry
        #     if category in business_data[entry]['categories']:
        #         row.append(1)
        #     # else if the category is not in the entry
        #     else:
        #         row.append(0)
        # remove stray text, such as "\n" form address
        # set up an array for the cleaned row entries
        row_clean = []
        # for every item in the row
        for item in row:
            # scan and replace for nasty text
            row_clean.append(str(item).replace('\n', ' '))
        # after all fields have been extracted and cleaned, append the row to the rows array for writing to csv
        rows.append(row_clean)

    for headers in business_header_additions:
        header.append(headers)
    length = len(header)
    print(length)
    # write to csv file
    # print(header)
    with open(output_path, 'w') as out:
        writer = csv.writer(out)
        # write the csv headers
        writer.writerow(header)
        # for each entry in the row array
        print('writing contents to csv...')
        for entry in tqdm(range(0, len(rows))):
            try:
                assert(len(rows[entry])==length)
                # write the row to the csv
                writer.writerow(rows[entry])
            # if there is an error, continue to the next row
            except UnicodeEncodeError:
                continue
    out.close()

# parse_data('data/train.json','data/business.csv')
parse_data('data/full_business.json','data/full_business.csv', 10914)
