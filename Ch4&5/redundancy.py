import reactionNetwork as rn 
# The full DCred network
DCred = rn.ReactionNetwork([
	([0,1],[0,0]), 		# x+y  -> x+x
	([0,1],[1,1]),		# y+x  -> y+y
	([0,3],[0,0]),		# x+y' -> x+x
	([0,3],[3,3]),		# x+y' -> y'+y'
	([2,1],[2,2]),		# x'+y -> x'+x'
	([2,1],[1,1]),		# x'+y -> x'+x'
	([2,3],[3,3]),		# x'+y' -> y' + y'
	([2,3],[2,2])		# x'+y' -> x'+x'
	], [1.0 for i in range(8)], [5,5,5,5],
	[lambda x,y : x[0]*x[1]*y, 
	lambda x,y, : x[0]*x[1]*y, 
	lambda x,y : x[0]*x[3]*y, 
	lambda x,y : x[0]*x[3]*y,
	lambda x,y : x[2]*x[1]*y, 
	lambda x,y : x[2]*x[1]*y, 
	lambda x,y : x[2]*x[3]*y,
	lambda x,y : x[2]*x[3]*y
	])

DCnoy = rn.ReactionNetwork([
	([0,1],[0,0]), 		# x+y  -> x+x
	([0,1],[1,1]),		# y+x  -> y+y
	([2,1],[2,2]),		# x'+y -> x'+x'
	([2,1],[1,1]),		# x'+y -> x'+x'
	], [1.0 for i in range(4)], [10,20,10],
	[lambda x,y : x[0]*x[1]*y, 
	lambda x,y, : x[0]*x[1]*y, 
	lambda x,y : x[2]*x[1]*y, 
	lambda x,y : x[2]*x[1]*y, 
	])

