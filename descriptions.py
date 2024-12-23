from ga_src.env import POP_SIZE, LENGTH, MAX_LS_ITER, INNER_LS_ITER

init_cvrp_learning_1 = '''
You will help me create initial high-quality solutions for the Capacitated Vehicle Routing Problems (CVRPs), which will make the subsequent search algorithms faster. CVRP is a problem that tries to find the best routes for a number of vehicles with limited space to deliver goods to various customers with different needs, while keeping the total transportation cost as low as possible. The total cost of CVRP depends on the exact locations of the customers, their demands, and the depot where the vehicles are based, subject to these common strict rules:
1) One vehicle must visit each customer only once.
2) The total demand of the customers on a vehicle's route must not exceed the vehicle's capacity (note that all vehicles have the same capacity).
3) Each route must start and end at the depot.
To create initial high-quality solutions for the new CVRPs, I will first show you some solved CVRPs as examples. You can use them to learn heuristics for building the routes, i.e., how the nodes are connected.
The following formats are used to provide the solved problems, where '{}' indicates variables:
<CVRP name={} n_customer={} capacity={}>
<Depot index_id=0 X-axis={} Y-axis={} demand=0></Depot>
<Customers>
    <customer index_id={} X-axis={} Y-axis={} demand={}></customer>
    ......
    <customer index_id={} X-axis={} Y-axis={} demand={}></customer>
</Customers>
<SOLUTION>
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
    ......
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
</SOLUTION>
</CVRP>

In this format, 'name', 'n_customer' and 'capacity' represent the name of the instance, the number of customers that need to visit and the capacity of each vehicle.
The second line shows the coordinates (i.e., X-axis and Y-axis) of the depot with index 0. Notably, the demand of the depot vertex is always 0
The attributes of all customers will be further outlined in '<Customer></Customer>' one by one, where 'X-axis' and 'Y-axis' present the coordinates and 'demand' denotes the occupied capacity of serving the customer.
Finally, the optimal serving routes with minimum total travel cost will be provided in '<Customer></Customer>' one by one, each of which contains the travel cost (specified by travel_cost), travel demand (specified by travel_demand) and a list of IDs of visited customers.

You will receive several solved CVRPs with the description of XML text in following multiple sessions, one solved CVRP per session.
'''

last_cvrp_learning_1 = '''
Now, you will be given several unsolved CVRPs in XML format, without the <SOLUTION>{}</SOLUTION> tag. Your task is help me to devise high-quality routes for each CVRP instance, based on the heuristics you learned from the solved CVRPs.

Note: you are required to return me the XML text with complete and entire routes without any explanations like:
<SOLUTION name={}>
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
    ......
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
</SOLUTION>
where the 'name' in 'SOLUTION' indicates the name of CVRP that you tackled and '{}' represent the variables that you should provide. Moreover, each customer has to be in one and exactly one route.
The descriptions of these CVRPs require to solve will be given in following multiple sessions.
'''

init_cvrp_learning_2 = '''
You will help me create initial high-quality solutions for the Capacitated Vehicle Routing Problems (CVRPs), which will make the subsequent search algorithms faster. CVRP is a problem that tries to find the best routes for a number of vehicles with limited space to deliver goods to various customers with different needs, while keeping the total transportation cost as low as possible. The total cost of CVRP depends on the exact locations of the customers, their demands, and the depot where the vehicles are based, subject to these common strict rules:
1) One vehicle must visit each customer only once.
2) The total demand of the customers on a vehicle’s route must not exceed the vehicle’s capacity (note that all vehicles have the same capacity).
3) Each route must start and end at the depot.

To produce initial high-quality solutions for the unseen CVRPs, I will first present you with some solved CVRPs as examples that you can learn 'heuristics' for constructing the routes of unseen CVRPs, i.e., relationships between the nodes.
More specifically, I will present a visual illustration accompanied with a XML description for each solved CVRP, featuring its intrinsic topological layout and its optimal traverse routes. 

The following format is used to describe each solved CVRP with text information, where {} denotes variables:
<CVRP name={} n_customer={} capacity={}>
<Depot index_id=0 X-axis={} Y-axis={} demand=0></Depot>
<Customers>
    <customer index_id={} X-axis={} Y-axis={} demand={}></customer>
    ......
    <customer index_id={} X-axis={} Y-axis={} demand={}></customer>
</Customers>
<SOLUTION>
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
    ......
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
</SOLUTION>
</CVRP>

In this format, 'name', 'n_customer' and 'capacity' represent the name of the instance, the number of customers that need to visit and the capacity of each vehicle.
The second line shows the coordinates (i.e., X-axis and Y-axis) of the depot with index 0. Notably, the demand of the depot vertex is always 0
The attributes of all customers will be further outlined in '<Customer></Customer>' one by one, where 'X-axis' and 'Y-axis' present the coordinates and 'demand' denotes the occupied capacity of serving the customer.
Finally, the optimal serving routes with minimum total travel cost will be provided in '<Customer></Customer>' one by one, each of which contains the travel cost (specified by travel_cost), travel demand (specified by travel_demand) and a list of IDs of visited customers

Now you will be provided with several solved CVRPs with the description of XML text and the figure with original topological layout and optimal traverse routes in following multiple sessions, one solved CVRP per session.
'''

last_cvrp_learning_2 = '''
Now you are presented with the unsolved CVRPs with the description of XML text (without <SOLUTION>{}</SOLUTION>) and topological layout picture (without the figure of optimal traverse routes). Please attempt to devise high-quality preliminary routes for each CVRP instance one by one based on the heuristics you learned from solved CVRPs.

Note: you are required to return me the XML text without any explanations like:
<SOLUTION name={}>
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
    ......
    <route id={} travel_cost={} travel_demand={}>[{**sequence of route**}]</route>
</SOLUTION>
where the 'name' in 'SOLUTION' indicates the name of CVRP that you tackled and '{}' represent the variables that you should provide. Here are the descriptions and maps of the CVRPs that need to be solved, one CVRP per session.
'''

failed_refine = '''
Your routing solution is invalid. Each customer must appear exactly once in the routes. To return valid routes, refine the ones below by removing duplicate customer IDs and adding missing ones.\n
'''

