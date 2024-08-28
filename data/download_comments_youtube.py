import googleapiclient.discovery
import googleapiclient.errors
import csv

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyBdzAfHlGUdkudF5jYAuGwHonDF1GwoezI"

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

video_ids = ["PvlhJPtEstA", "WDk0jy-CiLs", "YSoc5kR0RP8", "g1fdVs3uJnA", "xzIKCYl4z38", "c7b74Ex-kCA", "zPRIW50IYGU", "wNU59vgr6mk", "zpMKoC-B7w4", "_66eRcxv4Fk", "NE__Imj_Sv0", "zjR9BPAt-jo", 
             "pS9hHYITWVQ", "-cxDsZlfQQo", "j7gBiTH3DIg", "st4nul77_Oc", "D3_B8kKmUjc", "FtBcuwX8j_0", "J5Cg86mX9kc", "f6EPrj5Jjns", "8e1U-sfyaXs", "HxuCXIgilFc", "sBuYrv_W1hM", "PUyqWp5V3sE",
              "l2TYc4sf9dw", "d8eRvVLfsnI", "gG62Vu4qwsk", "9ivr3YoZmCo", "VJKFRs1xGe8", "0ujWdDWJ0Ck", "9j41I4OPFbY", "YhRDIGr3Ag0", "KHNPb81a21M", "DO6VjP2rIu8", "9K0R8kWCauo", "SDxqL7L2Ivo",
              "ul_GJAU6e-g", "R2K4IQGpiGM", "nVeCIbiouUk", "Y94GQrRkaGY", "2MYQBaog-d4", "XmB6OyA7iQU", "HrDU5SxepBA", "mYFcmLX4mUc", "JD-tAYIpgaA", 
              "-IUyAKMtBWc", "yhNUAnnIcMo", "gnq_bShXo60", "jOI5vUWJbPk", "y-5Vl61eZmY", "CMPqrA5suRo", "CMPqrA5suRo"]

def fetch_and_write_comments(video_id, csv_writer):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100, 
        textFormat="plainText"
    )
    while request:
        response = request.execute()

        for item in response['items']:
            author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            csv_writer.writerow([video_id, author, comment]) 

        request = youtube.commentThreads().list_next(request, response)

with open('youtube_comments.csv', 'a', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(['Video ID', 'Author', 'Comment'])

    for video_id in video_ids:
        fetch_and_write_comments(video_id, csv_writer)
