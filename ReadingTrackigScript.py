import pandas as pd
import glob
import numpy as np
#import math
import geopandas as gp
from shapely.geometry import Point, Polygon
import os

def main():
    #paths
    dataPath = "/Users/*****/Documents/*****/Data/PRE-LOCKDOWN-ALL-DATA/Preprocessing/Eye_Raw"
    outputPath= "/Users/*****/Documents/*****/*****/Reading/FINAL"
    timestampPath = '/Users/*****/Documents/*****/Data/PRE-LOCKDOWN-ALL-DATA/Preprocessing/BipsTimestamps/TasksTimestamps.csv'
    readingAOIPath = '/Users/*****/Documents/*****/AOIs/Reading/Reading_AOIs.csv'

    #read task timestamps    
    df_timeStamps = pd.read_csv(timestampPath)
    #read and create geopandas dataframe for the AOIs representing each word in the reading task 
    df_AOI = pd.read_csv(readingAOIPath)
    df_AOI['geometry'] = df_AOI.apply(lambda row: Polygon([(row.tl_x,row.tl_y),(row.tr_x,row.tr_y),(row.br_x,row.br_y),(row.bl_x,row.bl_y)]), axis=1)
    gdf_AOI = gp.GeoDataFrame(df_AOI[['word_id' , 'word' , 'geometry']])
    #fetch all TOBII fileNames 
    pid_files = glob.glob(dataPath + "/*.tsv")
    #initialize result dataframe
    data_all = pd.DataFrame([])
    for path in pid_files:
        print(path)
        patient = os.path.basename(path).replace('.tsv', '').split("_")[1]
        df_data = pd.read_csv(path,sep='\t')

        #keep ET data corresponding to Reading Task as marked by the corresponding "bip" markers
        df_patient_timeStamps = df_timeStamps[df_timeStamps['StudyID'] == patient]
        df_times = df_patient_timeStamps[df_patient_timeStamps['Task'] == 'Reading']
        initialTime = df_times['timestampIni_bip'].item()
        endTime = df_times['timestampEnd_bip'].item()

        initalIndex = df_data[df_data['RecordingTimestamp'] >= initialTime].index[0]
        endIndex = df_data[df_data['RecordingTimestamp'] > endTime].index[0]-1
        df_reading = df_data.loc[initalIndex : endIndex]
        #Drop duplicated and NA fixations
        df_reading = df_reading.drop_duplicates('FixationIndex', keep='first').dropna(subset=['FixationIndex'])
        #Create master dataframe with the row index of each fixation in the TOBII dataframe
        df_master = pd.DataFrame(df_reading['FixationIndex'].index)
        if(df_master.empty):
            print('Empty Dataframe', patient)
        else:
            #the row index will be called FixationIndex --not to be confused with the FixationID (which is called FixationIndex in the TOBII Dataframe)
            df_master = df_master.rename(columns={0: "FixationIndex"})
            
            #Add participantID
            df_master = pd.DataFrame(df_master.assign(PatientName = patient))
            
            #Fetch fixation iunfo from Tobii export
            df_master = df_master.assign(StartFixation = df_data.loc[df_master['FixationIndex'], ['RecordingTimestamp']].values, #fetch fixation timestamp
                                         StartFixationID = df_data.loc[df_master['FixationIndex'], ['FixationIndex']].values, #fetch fixationID (called FixationIndex in Tobii)
                                         StartFixationDuration = df_data.loc[df_master['FixationIndex'], ['GazeEventDuration']].values, #fectch fixation duration
                                         PointXStart = df_data.loc[df_master['FixationIndex'], ['FixationPointX (MCSpx)']].values, #fetch fixation x position
                                         PointYStart = df_data.loc[df_master['FixationIndex'], ['FixationPointY (MCSpx)']].values) #fetch fixation y position
            
            #create a geopandas dataframe in order to assign fixations to AOIS i.e. words in the text
            gdf_masterStart = gp.GeoDataFrame(df_master[['FixationIndex', 'PointXStart', 'PointYStart']]).fillna(-1)
            gdf_masterStart['geometry'] = gdf_masterStart.apply(lambda row: Point(row.PointXStart, row.PointYStart), axis=1)
            #merge geopandas dataframe with geopandas AOI dataframe to find hits (points falling on AOIS)
            gdf_masterStart = gp.sjoin(gdf_masterStart, gdf_AOI, how="left", op="within")
            #add word_id and word of each hit to master dataframe
            df_master['Word_ID_Start'] = gdf_masterStart['word_id']
            df_master['Word_Start'] = gdf_masterStart['word']
            
    
            #Given a startfixation at position i, the endfixation will be the fixation at position i+1
            #Compute all endFixation-related metrics by shifting sartFixation from 1 position
            df_master['EndFixation'] = df_master['StartFixation'].shift(-1)
            df_master['EndFixationID'] = df_master['StartFixationID'].shift(-1)
            df_master['EndFixationDuration'] = df_master['StartFixationDuration'].shift(-1)
            df_master['PointXEnd'] = df_master['PointXStart'].shift(-1)
            df_master['PointYEnd'] = df_master['PointYStart'].shift(-1)
            df_master['Word_ID_End'] = df_master['Word_ID_Start'].shift(-1)
            df_master['Word_End'] = df_master['Word_Start'].shift(-1)
            #delete last row of dataframe as the last saccade needs ends with the last valid fixationEnd
            df_master=df_master[:-1]
            
            #saccade duration
            df_master['SaccadeDuration'] = (df_master['EndFixation'] - df_master['StartFixation']
                                                                - df_master['StartFixationDuration'])
            #saccade length in terms of euclidean (x,y) distance
            df_master['Length'] = np.sqrt((df_master['PointXEnd'] - df_master['PointXStart']) ** 2 + (df_master['PointYEnd'] - df_master['PointYStart']) ** 2)
            #sacccade length in terms of words
            df_master['Length_Words'] = df_master['Word_ID_End'] - df_master['Word_ID_Start']
            #identify saccades as regressions "R" or saccades "S"
            df_master['Direction'] = np.where((df_master.Word_ID_End - df_master.Word_ID_Start < 0), 'R', 'S' )
            
            #Fetch only fields of interest for the result dataframe
            df_res = df_master[['PatientName', 'StartFixationID','StartFixation', 'EndFixationID', 'EndFixation',
                                'StartFixationDuration', 'EndFixationDuration',
                                'SaccadeDuration', 'Length', 'Length_Words', 'Direction', 
                                'Word_ID_Start','Word_Start','Word_ID_End', 'Word_End']]
            #reset index
            df_res.reset_index(drop=True, inplace=True)
            
            #In order to accurately compute first pass and first occurence, we find the minimum word_id in the first 10 -not NA- fixations. 
            #This corresponds to the first fixation that the participant did in order to start reading the text
            participant_starts_reading_index = df_res['Word_ID_Start'].dropna().iloc[0:9].idxmin()
            #we get rid of the initial fixations that do not constitute reading behavior
            df_res = df_res.iloc[participant_starts_reading_index:]
            
            #for each fixation at the beginning of a saccade, we compute whether it's a frist occurrence (first time the word_id is fixated)
            df_res['FirstOccurence'] = np.where((df_res.groupby('Word_ID_Start').cumcount() > 0), False, True )
            #for each fixation at the beginning of a saccade, we compute whether it's in the first pass (no later word_id has been fixated yet)
            df_res['FirstPass'] = np.where((df_res.Word_ID_Start >= df_res.Word_ID_Start.cummax() ), True, False )
            
            #concatenate participant to rest of participants
            data_all = data_all.append(df_res)
    #print features
    data_all.to_csv(outputPath + '/ReadingFeaturesExtras'+'.csv',encoding='utf-8')   
if __name__ == "__main__":
    main()
