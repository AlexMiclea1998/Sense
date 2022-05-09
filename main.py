import pandas as pd

from api.YoutubeApi import getYoutubeComments, getYoutubeStats

if __name__ == '__main__':
    link="MN8JBVfYD78"
    nrComments=100
    # comments=getYoutubeComments(link,nrComments)
    # print(comments)
    # stats=getYoutubeStats(link)
    # print(stats)
    # neutral = 0
    # positiv = 0
    # negativ = 0
    # data = pd.read_csv('misc/output.csv')
    # neutral=data.query('Label==0')
    # positiv = data.query('Label==1')
    # negativ = data.query('Label==-1')
    # positiv[1000:1400].to_csv('Comments.csv', mode='a')
    # negativ[:1000].to_csv('Comments.csv', mode='a')
    # for r in data:
    #     if r['Label']==0:
    #         neutral+=1
    #     elif r['Label']==1:
    #         positiv+=1
    #     elif r['Label']==-1:
    #         negativ+=1

    # print("pozitiv:",positiv.count())
    # print("negativ:", negativ.count())
    # print("neutral:", neutral[:1000])
    # data = pd.read_csv('Comments.csv')
    # data.drop(['Index'], axis=1).to_csv('Comments.csv')



