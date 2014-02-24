---
layout: default
---
#Some auction strategies
In online ad industry, ad inventory is increasingly sold through Real Time Bidding 
(RTB). In this article, we describe three of the most prominent auction strategies 
used in RTB. All these auctions are sealed bid auctions. (i.e. bidders do not know
about values bid by each other)

##Second price auction
This is an auction method where there is one object to be sold and multiple buyers.
Each buyer bids a value. The highest bidder is awarded the object and he pays the
price equal to the second highest bid value. For instance, if a pen is being auctioned
and P1, P2, P3 bid &#8377; 70, 50, 40 respectively then P1 wins the auction and
pays &#8377; 50.

This mechanism is used in auctioning display inventory by various ad exchanges
like [doubleclick] (https://support.google.com/adxbuyer/answer/152039?hl=en).

####Truth telling is dominant strategy in second price auction####

What it means is that each participant has an incentive to bid the true value that
he attaches to the object.

Let\'s see this using an example. Let\'s say an auction for a pen is taking place.
I value that pen at &#8377; 50. Then, it is best for me to bid &#8377; 50. Why?
* Suppose, I become aggressive and bid 60 instead of 50. Then
  + If highest bid other than mine was more than 60, then I would lose the bid.
    But I would have lost the bid even if I bid 50. So, truth telling is no worse
    than bidding 60.
  + If higest bid other than mine was between 50 and 60, I would end up winning the
    bid and paying more than 50. Since intrinsic value of the pen to is only 50,
    this is a unprofitable proposition for me. In this case, truth telling would
    have avoided this loss and was thus a better strategy.
  + If highest bid other than mine was less than 50, I would win the bid whether
    I bid 50 or 60, and pay same price irrespective of my bid. Hence, truth telling
    is no worse than bidding 60.
* Similarly, suppose I bid conservative and bid 40 instead of 50. Then
  + If highest bid other than mine is more than 50. I would lose the bid irrespctive
    of bidding 50 or 40. So, truth telling is no worse than bidding lower.
  + If highest bid other than mine is between 40 and 50, say 44then bidding 40 makes 
    me lose the bid, where has bidding 50 would have led me to win the bid, pay 44
    and make a profit of 4.
  + If higest bid other than mine is less than 40, then I would win the bid regardless
    of bidding 40 or 50.

As can be seen from above, truth telling is never worse off than bidding higher or
lower and sometimes it is better than alternative strategies. Hence, truth telling
is a dominant strategy in second price auction.

####Why auction systems where truth telling is dominant strategy are good####
In case participants are not incentivized to bid their true values, their bids would
be dependent on their guess about other participants' bids. This would lead to 
differing bid values by participants even when inventory quality remains the same.
Thus, revenues of seller would be variable, and that is not desirable by the sellers.

##Generalized second price (GSP) auction
Both GSP and VCG are auction stratgies where multiple items are for sale. There are
various ways to understand what GSP and VCG are. In this article, let's consider
the setting that 

##Vickry Clarke Groves (VCG) auction
