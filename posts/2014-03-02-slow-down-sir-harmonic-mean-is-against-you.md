---
layout: default
---
#Slow down, sir, harmonic mean is against you 
##A case against speeding

A few months ago, I was going from Bangalore to Thiruvannamalai. While most of
the roads in Tamilnadu are good, NH-66 connecting Krishnagiri and Thiruvannamalai 
has parts which are good, and other parts are muddy and often pass through 
populated areas.

In such a situation, the intuition is often to drive very fast on the good part
of the road, faster than what you would have done if the whole road were 
good. The intuition is to "make up" for the slow speed in the bad part of the
road. In the slow portion, we are bottlenecked by congestion on the road, so we cannot
do much. In the fast portion, we can be more aggressive and drive faster.

How much can we make up?

Let us say total distance to be travelled is 400km, of which 200km is good and
200km is bad. Speed on good road is 100km/hr and speed on bad road is 20km/hr.
What do you think is the average speed during the journey. First intuition
is that the average speed should be (100 + 20) / 2 = 60km/hr.

However, actually, the average speed is the harmonic mean of 100 and 20.

Time taken to cover fast part: 200 / 100 = 2 hrs

Time taken to cover slow part: 200 / 20 = 10 hrs

Total time: 12 hrs

Average speed: 400/12 km/hr = 33.3 km/hr = 2 * 100 * 20 / (100 + 20) km/hr.

Let's see how much does speeding buy us. We decide to drive at 20% higher speed
in the fast portion, at 120 km/hr. Then average speed for the journey would be
2 * 120 * 20 / (120 + 20) = 34.3 km/hr, a mere 3% speedup over 33.3km/hr.

If similar 20% speedup could be had for the slower part from 20km/hr to 24km/hr, 
gain would be much more handsome. Average speed up goes up to 2 * 100 * 24 / (100 + 24) = 38.7km/hr,
which is a handsome 16% speedup!

Reason is that most of the time is spent on the slow road, so it makes sense
to increase the speed there. Not doing so is falling prey to [Streetlight effect](http://en.wikipedia.org/wiki/Streetlight_effect)
##Moral of the story
Moral of the story is that it makes more sense to make modest speedups in slower
portion of the job rather than big speedups in faster portion. And speeding on the
road is not worth the risk.
