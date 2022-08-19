from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
     
    print("Overall sentiment dictionary is : ", sentiment_dict)
    if sentiment_dict["neg"]:
        return ("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    elif sentiment_dict["neg"] and ["neu"]:
         return ("sentence was rated as ", sentiment_dict['neg']*100, "% Negative and", sentiment_dict['neu']*100, "% Neutral")
    elif sentiment_dict["neg"] and ["neu"] and ['pos'] :
         return ("sentence was rated as ", sentiment_dict['neg']*100, "% Negative and", sentiment_dict['neu']*100, "% Neutral", sentiment_dict['pos']*100, "% Neutral")
    #print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    #print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    #print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
 
    #print("Sentence Overall Rated As", end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
            return("Positive")
    
    elif sentiment_dict['compound'] <= - 0.05 :
            return("Negative")
    
    else :
            return("Neutral") 
 
 
   
# Driver code
if __name__ == "__main__" :
 
    print("\n1st statement :")
    sentence = ("very bad i'm unhappy with the product\n")
    # function calling
    sentiment_scores(sentence)