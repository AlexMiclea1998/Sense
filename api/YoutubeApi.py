import os
import googleapiclient.discovery
import googleapiclient.errors
import csv
from textblob import TextBlob


os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCVI-_MDRtGsPosiY42YPOJyfRHCYxcF2o"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

def getYoutubeComments(videoId,nrComments):
    request = youtube.commentThreads().list(
        part="snippet,replies",
        maxResults=nrComments,
        videoId=videoId)
    response = request.execute()
    commentsBlocks = response['items']
    comments = []
    for i in commentsBlocks:
        comments.append(i['snippet']['topLevelComment']['snippet']['textDisplay'])

    # #print(comments)
    # with open('comments.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for comment in comments:
    #         writer.writerow([getPolarity(comment),comment])

    return comments



def getYoutubeStats(videoId):
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=videoId
    )
    response = request.execute()
    # return {'viewCount': '76825', 'likeCount': '1028', 'favoriteCount': '0', 'commentCount': '57'}
    return response['items'][0]['statistics']
