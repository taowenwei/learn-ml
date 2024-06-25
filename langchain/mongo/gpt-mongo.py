import mongo_chatbot
import mongo_toolkit

bot = mongo_chatbot.MongoChat()
toolkit = mongo_toolkit.MongoToolkit('sample_mflix')

schema = toolkit.getSchema('movies', {'title':
                                      'Sin City: A Dame to Kill For'})
bot.addSchema('movies', schema)
schema = toolkit.getSchema('comments', {'name':
                                       'Gregor Clegane'})
bot.addSchema('comments', schema)
bot.addKnowledge('''the `_id` field of the movies collection 
                 links to the `movie_id` field of the comments collection''')


bot.askQuestion('''use aggrgation, how to query a movie by title then find its comments''')
bot.askQuestion('''use aggrgation, how to get the number of movies released each year''')