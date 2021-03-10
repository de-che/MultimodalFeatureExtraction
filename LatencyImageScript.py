import pandas as pd
import glob
import numpy as np
import geopandas as gp
from shapely.geometry import Point, Polygon
from pandas.core.common import flatten
import os

def myPoint(n):
    return Point(n)

def main():
    #PATH FOR INPUT FILES HERE
    path1 = "/Users/*****/Desktop/*****-Work/Scripts/input_files" 
    all_files = glob.glob(path1 + "/*.csv")

    #We iterate through every file and read it
    for file in all_files:
        path = file
        patient = os.path.basename(path).replace('.csv', '')
        df = pd.read_csv(path)
        dfAOI = df[df["InfoUnit"].notna()]
        #WINDOWS ARE DEFINED HERE
        backWindow = 2000
        frontWindow = 2000
        # We create a marker on the dataframe with only AOI'S to mark the difference of consecutive groups of AOI'S
        dfAOI['marker'] = (dfAOI['InfoUnit'] != dfAOI['InfoUnit'].shift()).cumsum()
        # We keep the first and last indices of each group and we add them as columns to our new df_master, we also reset the index
        df_master = dfAOI.index.to_series().groupby(dfAOI['marker']).agg(['first','last']).reset_index()
        df_first = df_master[['first']]
        df_last = df_master[['last']]
        #We add our patient to a new patient coulmn
        df_master['PATIENT'] = patient
        # We query the main dataframe to find the recording time stamp of our first and last indices 
        df_firstIndexTime = df.loc[df_master['first'], ['RecordingTimestamp']]
        df_lastIndexTime = df.loc[df_master['last'], ['RecordingTimestamp']]
        #We assing the target time and the recorind timestamp start and end in our df_master
        df_master= df_master.assign(recordingTimeStampStart=df_firstIndexTime.values,recordingTimeStampEnd=df_lastIndexTime.values)
        df_master=df_master.assign(timeTargetBack=df_master['recordingTimeStampStart'].values-backWindow,timeTargetFront=df_master['recordingTimeStampEnd'].values+frontWindow)
        #We keep a copy of the index for future use 
        df['copy_index'] = df.index
        # We use merge_asof to find the closes recording timestamp from forward and backward directions
        df_backward = pd.merge_asof(df_master, df, left_on='timeTargetBack',right_on='RecordingTimestamp',direction='backward')
        df_forward = pd.merge_asof(df_master,df,left_on='timeTargetFront',right_on='RecordingTimestamp',direction='forward')
        #We take results of the merfe asof and use the copy_index to keep the correct index
        df_forward[['RecordingTimestamp', 'timeTargetBack','copy_index']]
        #Assing to everything to df_master
        df_master=df_master.assign(backWindowIndex=df_backward['copy_index'].values)
        df_master=df_master.assign(frontWindowIndex=df_forward['copy_index'].values)
        df_master=df_master.assign(InfoUnits=df.loc[df_master['first'],'InfoUnit'].values)
        df_list = []
        df['point'] = list(zip(df['FixationPointX..MCSpx.'], df['FixationPointY..MCSpx.']))
        #Create smaller dataframes from the main dataframe 
        for index, row in df_master.iterrows():
            df_list.append(df.loc[row['backWindowIndex'] : row['frontWindowIndex']].drop_duplicates('FixationIndex'))
        
        #Define polygons
        polys = gp.GeoSeries({
            'BOY': Polygon([(590,165),(680,200),(605,655),(495,617),(510,300),(595,170)]),
            'JAR': Polygon([(400,115),(520,115),(520,235),(400,235)]),
            'COOKIE': Polygon([(420,260),(475,260),(475,320),(420,320)]),
            'STOOL': Polygon([(495,617),(605,655),(522,878),(438,869),(386,788)]),
            'GIRL': Polygon([(255,444),(452,350),(413,739),(386,788),(420,845),(420,900),(325,910)]),
            'WOMAN': Polygon([(920,145),(1020,145),(1045,350),(985,395),(990,455),(950,500),(990,565),(1040,560),(1065,670),(1010,880),(890,890),(830,550)]),
            'PLATE': Polygon([(1045,350),(1110,365),(1110,415),(1060,460),(990,455),(985,395)]),
            'DISHCLOTH': Polygon([(1060,460),(1040,560),(990,565),(950,500),(990,455)]),
            'CURTAINS': Polygon([(970,145),(970,90),(1480,90),(1480,615),(1385,575),(1425,445),(1285,230),(1250,120),(1210,230),(1080,355),(1045,350),(1020,145)]),
            'WINDOW': Polygon([(1250,120),(1285,230),(1425,445),(1385,575),(1055,510),(1060,460),(1110,415),(1110,365),(1089,355),(1210,230)]),
            'SINK': Polygon([(1055,510),(1350,565),(1222,672),(1055,605),(1040,560)]),
            'WATER': Polygon([(1222,672),(1155,990),(890,990),(830,940),(890,890),(1010,880),(1065,670),(1055,605)]),
            'DISHES': Polygon([(1300,625),(1480,625),(1480,710),(1300,710)])
            })
        hitsList = []
        # Iterate throught the smaller dataframes and check which polygons it falls inside
        for x in df_list:
            pointList = list(map(myPoint, (x['point'].values)))
            _pnts = pointList
            pnts = gp.GeoDataFrame(geometry=_pnts)
            hitsList.append(pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()}))
        resList = []
        for data in hitsList:
            res = list(flatten(pd.DataFrame(data.columns.where(data == True).tolist()).values.tolist()))
            ans = [x for x in res if not isinstance(x, float)]
            noDuplicate = pd.Series(ans).drop_duplicates()
            resList.append(list(noDuplicate))
        hitName = 'HITS' + '_' + str(backWindow)
        df_master[hitName] = resList
        df_master
        hitToCSV = df_master[['PATIENT', 'InfoUnits', hitName]]
        df_backLatency = df_master
        df_backLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='KITCHEN'].index, inplace=True)
        df_backLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='EXTERIOR'].index, inplace=True)
        df_backLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='CUPBOARD'].index, inplace=True)

        df_listBackLatency = []
        df_backIndices = []
        for index, row in df_backLatency.iterrows():
            sub_df = df.loc[0 : row['first'] - 1].drop_duplicates('FixationIndex')
            pointList = list(map(myPoint, (sub_df['point'].values)))
            _pnts = pointList
            pnts = gp.GeoDataFrame(geometry=_pnts)
            df_table = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
            exactPoly =  df_table[['geometry', row['InfoUnits']]]
            lastHit = exactPoly.where(exactPoly == True).last_valid_index()
            if(lastHit != None):
                res = df.loc[row['first']]['RecordingTimestamp'] - sub_df.iloc[lastHit]['RecordingTimestamp'] 
                df_backIndices.append(sub_df.iloc[lastHit]['copy_index'])   
            else:
                df_backIndices.append(-1)
                res = -1
            df_listBackLatency.append(res)
        df_master['BackLatency'] = df_listBackLatency
        df_master['BackIndex'] = df_backIndices
        
        df_frontLatency = df_master
        df_frontLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='KITCHEN'].index, inplace=True)
        df_frontLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='EXTERIOR'].index, inplace=True)
        df_frontLatency.drop(df_backLatency.loc[df_backLatency['InfoUnits']=='CUPBOARD'].index, inplace=True)

        df_listFrontLatency = []
        df_frontIndices = []
        for index, row in df_frontLatency.iterrows():
            sub_df = df.loc[row['last'] + 1 : len(df)].drop_duplicates('FixationIndex')
            pointList = list(map(myPoint, (sub_df['point'].values)))
            _pnts = pointList
            pnts = gp.GeoDataFrame(geometry=_pnts)
            df_table = pnts.assign(**{key: pnts.within(geom) for key, geom in polys.items()})
            exactPoly =  df_table[['geometry', row['InfoUnits']]]
            firstHit = exactPoly.where(exactPoly == True).first_valid_index()
            if(firstHit != None):
                res = sub_df.iloc[firstHit]['RecordingTimestamp'] - df.loc[row['last']]['RecordingTimestamp']
                df_frontIndices.append(sub_df.iloc[firstHit]['copy_index'])   
            else:
                df_frontIndices.append(-1)
                res = -1
            df_listFrontLatency.append(res)
        df_master['FrontLatency'] = df_listFrontLatency
        df_master['FrontIndex'] = df_frontIndices

        latencyToCSV = df_master[['PATIENT', 'InfoUnits', 'BackLatency', 'FrontLatency']]

        patientResult = pd.merge(latencyToCSV, hitToCSV, right_index=True, left_index=True)

        patientResult.reset_index()
        resultsToCSV = patientResult.drop(['PATIENT_y','InfoUnits_y'], axis=1)
        resultsToCSV = resultsToCSV.rename(columns={'PATIENT_x': 'Patient', 'InfoUnits_x': 'InfoUnits'})
        mypath= "/Users/*****/Desktop/*****-Work/Scripts/Output"
        resultsToCSV.to_csv(mypath + '/' + patient + 'output' +'.csv',encoding='utf-8')

    allOutputs = []
    path = "/Users/*****/Desktop/*****-Work/Scripts/Output"
    all_files = glob.glob(path + "/*.csv")
    for file in all_files:
        allOutputs.append(pd.read_csv(file))
    df = pd.concat(allOutputs)
    df = df.reset_index(drop=True)
    df = df.drop(['Unnamed: 0'], axis=1)
    df.to_csv('MultimodalFeatures'+'.csv',encoding='utf-8')    

if __name__ == "__main__":
    main()