---
layout: default
---
#Projections in mongodb
In MongoDB, one should use [projections](https://docs.mongodb.org/manual/tutorial/project-fields-from-query-results/) wherever possible. I wanted to quantify two things

* What is the effect of using projections on query times?
* Why do query times improve because of projections?

We have a table called events, whose stats looks like this:

<pre><code>
> db.events.stats()
{
	"ns" : "6ed11a9246404d1b95fe.events",
	"count" : 16912712,
	"size" : 8617342080,
	"avgObjSize" : 509,
	"numExtents" : 25,
	"storageSize" : 9305935856,
	"lastExtentSize" : 2146426864,
	"paddingFactor" : 1,
	"paddingFactorNote" : "paddingFactor is unused and unmaintained in 3.0. It remains hard coded to 1.0 for compatibility only.",
	"userFlags" : 1,
	"capped" : false,
	"nindexes" : 4,
	"totalIndexSize" : 1167933424,
	"indexSizes" : {
		"_id_" : 548732240,
		"notificationId_1" : 14545104,
		"parameters.notificationId_1" : 54296816,
		"eventName_1" : 550359264
	},
	"ok" : 1
}
</code></pre>

So, there are around 17M entries, having a total size of 8.6GB, thus having an average object
size of 8.6GB/17M = 509 bytes.

I fetched first three million entries in this collection, using a python program like this:
<pre>
<code>
class QGMongo(object):
   __conn = None
   @classmethod
   def get_connection(cls):
      if cls.__conn is None:
         cls.__conn = MongoClient('127.0.0.1', 27000)
      return cls.__conn

if __name__ == '__main__':
   conn = QGMongo.get_connection()
   database = conn['6ed11a9246404d1b95fe']
   events = database.events.find()
   count = 0
   for event in events:
      count += 1
      if count == 3000000:
         break
</code>
</pre>

This prgram took around 120s to run.
I set db.setProfilingLevel(2) and then figured out from system.profiles collection that

As per db.system.profile collection,

* There were 273 getmore commands.
* They returned a total of 3M documents of length around 1.1GB in a total of 15s

I then did same experiment using projection:
<pre>
<code>
class QGMongo(object):
   __conn = None
   @classmethod
   def get_connection(cls):
      if cls.__conn is None:
         cls.__conn = MongoClient('127.0.0.1', 27000)
      return cls.__conn

if __name__ == '__main__':
   conn = QGMongo.get_connection()
   database = conn['6ed11a9246404d1b95fe']
   events = database.events.find({}, {'eventName': 1})
   count = 0
   for event in events:
      count += 1
      if count == 3000000:
         break
</code>
</pre>

This time, the program took around 30s. 

As per db.system.profile collection,

* There were 40 getmore commands
* They returned a total of 3M documents of length around 159MB in 1.7s. 

As a result of this, I have the answer to my first question: the query times improve significantly
using projections. When I used projection such that the data size required by my program dropped to
around 15% (1.1GB to 160MB), program running time dropped to 25% (120s to 30s).

However, my second question is not well answered. There are two outstanding questions:

* While complete running times for my programs in the two cases are 120s and 30s repectively,
times reported by mongodb are 15s and 1.7s respectively. So, where does the remaining time
get spent? Is it in the pymongo driver?  

* Why does projection help at all? The data required to be read from the disk is the
same. I have read somewhere that decoding bson is what takes time. I need to quantify
this time before I can come to a conclusion.
