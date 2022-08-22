from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# this is an exact replica of the other version used.
# This one is used when building on jenkins
def ScoresofSentiment(sentenceAnalysis):
 
    # Now, we create an object, a sentiment Intensity Analyzer
    SentimentIntensityObject = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentimentDictionary= SentimentIntensityObject.polarity_scores(sentenceAnalysis)
     
    print("Global analysis is the following :", sentimentDictionary)
    print("The input sentence has a rating of", sentimentDictionary['neg']*100, "% Negative")
    print("The input sentence has a rating of", sentimentDictionary['neu']*100, "% Neutral")
    print("The input sentence has a rating of", sentimentDictionary['pos']*100, "% Positive")
 
    print("Overall, the sentence is labeled as", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentimentDictionary['compound'] >= 0.05 :
            print("Positive")
    
    elif sentimentDictionary['compound'] <= - 0.05 :
            print("Negative")
    
    else :
            print("Neutral") 
 
 
   
# Driver code
if __name__ == "__main__" :
 
    print("\nthe first sentence has the following results :")
    sentenceAnalysis = ("very bad i'm unhappy with the product\n")
    ScoresofSentiment(sentenceAnalysis)

    print("\nthe second sentence has the following results :")
    sentence = ("I love and enjoy this product\n")
    # function calling
    ScoresofSentiment(sentenceAnalysis)