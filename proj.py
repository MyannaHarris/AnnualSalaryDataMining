#!/usr/bin/python
'''
  Myanna Harris
  Data Mining
  CPSC 310 - 01

  12-15-16
  proj.py

  Dataset: 
	annual salary information for all active employees of 
	Montgomery County, MD paid in 2015

  Creating training and test sets:
	Stratified

  Classifiers:
	K-nn
	Naive Bayes
	Decision Tree
	Random Forrest

  Predicting:
	Gender (1)
	Department (5/6)

  Using attributes:
        Gender (1)
	Department (5/6) (5 is the acronym)
	Annual Salary (2) (with a binning technique)
	2015 Gross Pay Recieved (3) (with a binning technique)
	Assignment-category (full or part -time)(7)
	Position Title (8)
	
  Deal with:
	Empty cells
	Can skip or take average
	Will try both

  To run:
  python proj.py 
  in directory where Employee_Salaries_2015.csv resides
'''

import matplotlib.pyplot as pyplot
from matplotlib.pyplot import cm
import math
import random
import tabulate
import numpy
import pylab
from random import shuffle

###### FUNCTIONS FROM PREVIOUS HOMEWORKS ######

#function from hw1
# replace missing values (NA) with meaningful average based on classifier
def replaceMissingWMeaningfulAvg(rows, newFile, attr, class_idx):
    out = []
    keys = attr.keys()
    for i in range(0,len(keys)):
        key = keys[i]
        val = attr[key]
        for r in rows:
            if r[key] == "NA":
                meaningfulList = [float(r2[key]) for r2 in rows if (r2[key]!="NA"and r[6] == r2[6])]
                r[key] = str(int(getAvg(meaningfulList)))
    for r in rows:
        out.append(",".join(r))
    file = open(newFile, "w")
    file.write("\n".join(out))
    file.close()

#function from hw1
# Get average of a list of numbers
def getAvg(list):
    total = 0.0
    for x in list:
        total += float(x)
    return total/(len(list)*1.0)

#function from hw2
# read in a csv file into a list of lists
def csvToTable(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\n"))
    file.close()
    table = []
    for row in rows:
        splitRow = row.split(",")
        table.append(splitRow)
    return table
    
# turn csv into table skipping title line
#function from hw4
def csvToTableSkipTitle(fileName):
    file = open(fileName, "r")
    rows = filter(None, file.read().split("\r"))
    file.close()
    table = []
    for i in range(1, len(rows)):
        splitRow = rows[i].split(",")
        table.append(splitRow)
    return table

#function from hw2
# get column from table
def getCol(table, index):
    return [ row[index] for row in table ]

# gets random subsamples  
def randomSubsample(table):
    tableTrain = []
    tableTest = []
    trainIdx = random.sample(range(0, len(table)), len(table)*2/3)
    for i in range(0, len(table)):
        if i in trainIdx:
            tableTrain.append(table[i])
        else:
            tableTest.append(table[i])
    return (tableTrain, tableTest)
    
# gets statistically accurate sub samples
def stratifiedSubsamplesGen(table, k, classIdx):
    subSamples = [[] for i in range(0, k)]
    classList = {}
    for row in table:
        classRank = row[classIdx]
        if not classList.has_key(classRank):
            classList[classRank] = []
        classList[classRank].append(row)
    
    for key in classList.keys():
        classTable = classList[key]
        num = len(classTable) / k
        extra = len(classTable) % k
        start = 0
        for x in range(0, k):
            end = start + num
            if x == k-1:
                end += extra
            for z in range(start, end):
                subSamples[x].append(classTable[z])
            start += num
    return subSamples

# accuracy
# correct hits divided by total hits
def calcAccuracy(confusionMatrix, size):
    trues = 0
    total = 0
    for i in range(0, size):
        trues += confusionMatrix[i][i]
        total += sum(confusionMatrix[i])
    return trues / (total * 1.0)

# error rate
def calcErrorRate(accuracy):
    return 1.0 - accuracy

# From hw 2 as a base
# bar chart
# lstIn = column of data to graph as a list
def makeBarChart(lstIn, filename, title, xlabel):
    lst = sorted(lstIn)
    
    x = []
    y = []
    for v in lst:
        if v not in x:
            x.append(v)
            y.append(1)
        else: 
            y[len(y)-1] += 1
    xrng = numpy.arange(len(x))
    pyplot.bar(xrng, y, 0.45, align='center')
    
    pyplot.xticks(xrng, x)
    pyplot.xticks(rotation=70)
    pyplot.xticks(fontsize=8)
    
    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel('Count')
    if max(y)%10 == 0:
        yrng = numpy.arange(0, max(y)+11, 10)
        pyplot.yticks(yrng, yrng)
        
    
    pyplot.grid(True)
    pyplot.savefig(filename)
    pyplot.figure()

# From hw 2 as base
# multi-frequency diagram
# table = instances
# title = title of graph
# idx_label = the x-axis bins
# idx_cat = the colored bar divisions (classifiers)
# xlabel = label for x-axis
# ylabel = label for y-axis
# fileName = name for file the graph will be saved in
# legend = boolean, true = legend on page is normal size,
#           false = legend on page is smaller to fit all information
def makeMultFreq(table, title, idx_label, idx_cat, xlabel, ylabel, fileName, legend):
    # x-axis bin labels
    labels = [r[idx_label] for r in table]
    # make x-axis bin labels only store unique ones
    labels = sorted(list(set(labels)))
    # colored bar divisions (classifiers)
    labels_xs = list(set([r[idx_cat] for r in table]))

    # Fix some long labels for printing
    printLabels = ["" for i in range(0, len(labels))]
    for i in range(0, len(labels)):
        tempLabelTextList = labels[i].split("-")
        printLabels[i] = tempLabelTextList[0]

    # Divide instances into classifier divisions
    tables = [[] for k in range(len(labels_xs))]
    for r in table:
        for i in range(0, len(labels_xs)):
            if r[idx_cat] == labels_xs[i]:
                tables[i].append(r)

    # create a figure and one subplot
    # returns figure and x-axis
    fig, ax = pyplot.subplots()
    # create two bars (returns rectangle objects)
    xs = numpy.arange(1,len(labels)+1,1)

    # save colors and subplot bars
    # 38 colors
    #colors = ['black','lightgrey','blue', 'green', 'red', 'cyan',
                #'magenta','yellow','white', 'orange', 'beige', 'chocolate',
                #'darkgreen', 'fuchsia', 'gold', 'hotpink', 'lavender', 'lime',
                #'navy', 'palegreen', 'salmon', 'silver', 'slategray', 'olive',
                #'aliceblue', 'aquamarine', 'blanchedalmond',
                #'crimson', 'darkmagenta', 'darkslateblue',
                #'deeppink', 'forestgreen', 'ivory', 'lightblue',
                #'mintcream', 'mediumvioletred', 'papayawhip', 'rosybrown']
    colors = cm.rainbow(numpy.linspace(0,1,len(labels_xs)))
    rs = []

    # create subplot bars
    for t in range(0, len(tables)):
        vals = [ 0 for i in range(len(labels)) ]
        for row in tables[t]:
            index = labels.index(row[idx_label])
            vals[index] += 1
        rs.append(ax.bar([n-0.3 for n in xs], vals,
                         0.2, color=colors[t], label=labels_xs[t]))

    # create a legend, location upper left
    if legend:
        pyplot.legend(loc="upper left", bbox_to_anchor=[0, 1],
               ncol=2, shadow=True, title="Legend", fancybox=True)
    else:
        # Make legend smaller to fit
        pyplot.legend(loc="upper left", bbox_to_anchor=[0, 1],
               ncol=2, shadow=True, title="Legend", fancybox=True,
                      fontsize=6)
        
        
    
    # set x value labels
    pyplot.xticks(numpy.arange(1-0.2,len(labels)+1-0.2,1),printLabels)
    pyplot.xticks(rotation=70)
    pyplot.xticks(fontsize=8)

    pyplot.title(title)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.grid(True)
    pyplot.savefig(fileName)

# HW 3
# K-NN
# table to compare to
# instance to compare
# columns to use to calculate distance
# 1 = printing predictiona and actual, 0 = not printing
def fiveNearestNeighbors(table,instance,cols,isPrinting, class_idx):
    if isPrinting:
        print("instance: "+", ".join(instance))
    results = [[300000,1]]*5
    for row in table:
        if row != instance:
            distanceSquared = 0
            for col in cols:
                if isint(instance[col]):
                    normalizedDistance = findNormalizedDistance(
                        getCol(table,col),int(row[col]),int(instance[col]))
                else:
                    normalizedDistance = findNormalizedDistanceDiscrete(
                        row[col],instance[col])
                distanceSquared += pow(normalizedDistance,2)
            distance = math.sqrt(distanceSquared)
            checkResults(results,distance,row[class_idx])
    prediction = chooseNearest(results)
    actual = instance[class_idx]
    if isPrinting:
        print("class: "+str(prediction)+", actual: "+str(actual))
    return (prediction,actual)

# HW 3
# K-NN
# chooses knn result
def chooseNearest(results):
    options = {}
    for r in results:
        if r[1] not in options.keys():
            options[r[1]] = 1
        else:
            options[r[1]] += 1

    mostProb = 0
    nearestClass = ""
    for k in options.keys():
        if options[k] > mostProb:
            mostProb = options[k]
            nearestClass = k
    return nearestClass

# HW 3
# K-NN
# checks if row is in k nn
def checkResults(results,distance,val):
    if distance < results[-1][0]:
        place = len(results)-1
        while distance < results[place-1][0] and place > 0:
            place -= 1
        results.insert(place,[distance,val])
        results.pop()

# HW 3
# K-NN
# normalized distance
def findNormalizedDistance(col,val1,val2):
    maxVal = 6
    minVal = 1
    normalizedVal1 = (val1-minVal)/float(maxVal-minVal)
    normalizedVal2 = (val2-minVal)/float(maxVal-minVal)
    return normalizedVal1 - normalizedVal2

# based on HW 3
# K-NN
# normalized distance for discrete values
def findNormalizedDistanceDiscrete(val1,val2):
    if val1 == val2:
        return 0
    else:
        return 1

# HW 4
# Naive Bayes
# returns list of unique classifiers 
# and dictionary of each class table
def getUniqueClasses(table, classIdx):
    classLst = []
    classTables = {}
    
    for row in table:
        classCurr = row[classIdx]
        if classCurr not in classLst:
            classLst.append(classCurr)
            classTables[classCurr] = [row]
        else:
            classTables[classCurr].append(row)
            
    return (classLst, classTables)

# HW 4
# Naive Bayes
# table = table of data
# classIdx = index of class column
# classList = list of unique classifiers
# returns dictionary of classes and probabilities
def getClassProbabilities(table, classIdx, classList):
    probDict = {}
    idxDict = {}
    oppDict = {}
    i = 0
    
    for classifier in classList:
        probDict[classifier] = 0
        idxDict[classifier] = i
        oppDict[i] = classifier
        i += 1
    
    totalRows = len(table)
    for row in table:
        classCurr = row[classIdx]
        probDict[classCurr] += 1
        
    for key in probDict.keys():
        probDict[key] /= float(totalRows)
    
    return (probDict, idxDict, oppDict)

# HW 4
# Naive Bayes
# compares naive bayes prediction to actual
def naiveBayesComparison(row, classIdx,classProbs,classTbls, atts, isPrinting):
    if isPrinting:
        print("instance: "+", ".join(row))
    
    naiveBayesProbs = getNaiveBayesProbs(row, classProbs, classTbls, atts)
    
    mostProb = ""
    largestProb = -1
    for key in naiveBayesProbs.keys():
        if naiveBayesProbs[key] > largestProb:
            mostProb = key
            largestProb = naiveBayesProbs[key]
    
    prediction = mostProb
    actual = row[classIdx]
    
    if isPrinting:
        print("class: "+str(mostProbRank)+", actual: "+str(actRank))
    return (prediction,actual)

# HW 4
# Naive Bayes
# returns dictionary of classes
# and their probabilities for the row
def getNaiveBayesProbs(row, classProbs, classTables, atts):
    naiveBayesProbs = {}
    
    for key in classTables.keys():
        naiveBayesProbs[key] = 1
        
        totalRows = len(classTables[key])
        attDict = {}  
        for classRow in classTables[key]:
            for att in atts:
                if att not in attDict.keys():
                    attDict[att] = 0
                if row[att] == classRow[att]:
                    attDict[att] += 1
        for att in atts:
            attDict[att] /= float(totalRows)
            naiveBayesProbs[key] *= attDict[att]
        
        naiveBayesProbs[key] *= classProbs[key]
    
    return naiveBayesProbs

# HW 5
# Decision Tree
def tdidt(instances, atts, domains, classIndex):
    if len(domains) == 0:
        leaves = ['Leaves']
        leaves += partitionStats(instances, classIndex)
        return leaves
    elif len(domains) == 1:
        if sameClass(instances, classIndex):
            total = len(table)
            return ['Leaves',[instances[0][classIndex],total,total,1.0]]
        else:
            partitions = partitionInstances(instances, atts[0], domains[0])
            
            subtree = [atts[0]]
            for key, value in partitions.items():
                valueList = [key]
                if len(value) < 1:
                    valueList.append(tdidt(instances, [], [], classIndex))
                else:
                    valueList.append(tdidt(value, [], [], classIndex))
                subtree.append(valueList)
            return subtree
    else:
        partitionAttIdx = selectAttribute(instances, atts, classIndex)
        partitions = partitionInstances(instances, atts[partitionAttIdx], domains[partitionAttIdx])
        
        subtree = [atts[partitionAttIdx]]
        for key, value in partitions.items():
            valueList = [key]
            if len(value) < 1:
                valueList.append(tdidt(instances, [], [], classIndex))
            else:
                valueList.append(tdidt(value, 
                    [atts[i] for i in range(0,len(atts)) if i != partitionAttIdx], 
                    [domains[i] for i in range(0,len(domains)) if i != partitionAttIdx], classIndex))
            subtree.append(valueList)
        return subtree

# HW 5
# Decision Tree
def sameClass(instances, classIndex):
    classValue = instances[0][classIndex]
    for instance in instances:
        if instance[classIndex] == classValue:
            return False
    return True

# HW 5
# Decision Tree
def partitionStats(instances, classIndex):
    total = len(instances)
    partitionDict = {}
    stats = []
    info = getCol(instances, classIndex)
    for val in info:
        partitionDict.setdefault(val,0)
        partitionDict[val] = partitionDict[val] + 1
    for key, val in partitionDict.items():
        stats.append([key,val,total, float(val)/total])
    return stats

# HW 5
# Decision Tree
def partitionInstances(instances, attIndex, attDomains):
    partitioned = {}
    
    for attDom in attDomains:
        partitioned.setdefault(attDom, [])
    for i in instances:
        partitioned[i[attIndex]].append(i)
    return partitioned
        
# HW 5
# Decision Tree
def selectAttribute(instances, attIndexes, classIndex):
    Enews = [calcEnew(instances,attIndex,classIndex) for attIndex in attIndexes]
    return Enews.index(min(Enews))

# HW 5
# Decision Tree
def attFreqs(instances, attIndex, classIndex):
    attVals = list(set(getCol(instances, attIndex)))
    classVals = list(set(getCol(instances,classIndex)))
    result = {v:[{c:0 for c in classVals},0] for v in attVals}
    for row in instances:
        label = row[classIndex]
        attVal = row[attIndex]
        result[attVal][0][label] +=1
        result[attVal][1] += 1
    return result

# HW 5
# Decision Tree
def calcEnew(instances, attIndex, classIndex):
    results = attFreqs(instances, attIndex, classIndex)
    Enew = 0
    D=len(instances)
    for key,result in results.items():
        Dj = float(result[1])
        att = result[0]
        EDj = 0
        for K, attVal in att.items():
            value = (float(attVal) / Dj)
            EDj -= (value * math.log((value if value > 0 else 1),2))
        Enew += (float(Dj) / float(D)) * EDj
    return Enew

# HW 5
# Decision Tree
def tdidtClassifier(tree, instance, classIdx, isPrinting):
    predictedClass = getClassification(tree, instance)
    
    prediction = predictedClass
        
    actual = instance[classIdx]
    if isPrinting:
        print("class: "+prediction+", actual: "+actual)
    return (prediction,actual)

# HW 5
# Decision Tree
def getClassification(tree, instance):   
    idx = tree[0]
    
    if idx == 'Leaves':
        largestClass = ""
        largestProb = 0
        for i in range(1,len(tree)):
            if tree[i][3] > largestProb:
                largestProb = tree[i][3]
                largestClass = tree[i][0]
        return largestClass
    else:
        for i in range(1,len(tree)):
            if tree[i][0] == instance[idx]:
                return getClassification(tree[i][1], instance)

# HW 5
# Decision Tree
def printRules(tree):
    print("Rules: ")
    queue = []
    printTree(tree, queue)

# HW 5
# Decision Tree    
def printTree(tree, queue):
    if len(tree) < 2:
        return ""
        
    idx = tree[0]
    
    if idx == 'Leaves':
        largestClass = ""
        largestProb = 0
        for i in range(1,len(tree)):
            if tree[i][3] > largestProb:
                largestProb = tree[i][3]
                largestClass = tree[i][0]
        print "IF att1 = " + str(queue[0]),
        for i in range(1, len(queue)):
            print "and att"+str(i+1)+" = " + str(queue[i]),
        
        print "THEN class = " + str(largestClass)
    else:
        for i in range(1,len(tree)):
            queue.append(tree[i][0])
            printTree(tree[i][1], queue)
            del queue[-1]

# hw 6
# Random Forest
# Get randomly stratified test and remainder sets
def randomStratify(table, class_idx):
    test = []
    remainder = []
    class_tables = {}
    
    for row in table:
        if row[class_idx] not in class_tables.keys():
            class_tables[row[class_idx]] = []
        class_tables[row[class_idx]].append(row)
        
    for k in class_tables.keys():
        shuffle(class_tables[k])
        
    for k in class_tables.keys():
        oneThird = int(len(class_tables[k]) / 3)
        test = test + class_tables[k][0:oneThird+1]
        remainder = remainder + class_tables[k][oneThird:]
    return (test, remainder)

# hw 6
# Random Forest
# make N trees        
def createRandomForests(table, class_index, N, F, numClass, idxDict, atts, domains):
    forests = []
    for x in range(0,N):
        test, training = bootstrap(table)
        tree = randomTdidt(training, atts, domains, class_index, F)
        accuracy = determineAccuracy(tree,test, class_index, numClass, idxDict)
        forests.append([tree,accuracy])
    return forests

# hw 6
# Random Forest
# turn remainder into random test and train    
def bootstrap(table):
    test = []
    train = []
    idxs = [random.randint(0, len(table)-1) for _ in table]
    for i in range(0, len(table)):
        if i in idxs:
            train.append(table[i])
        else:
            test.append(table[i])
    return (test, train)

# hw 6
# Random Forest
# make decision tree using random attribute subset at nodes    
def randomTdidt(instances, atts, domains, classIndex, F):
    if len(domains) == 0:
        leaves = ['Leaves']
        leaves += partitionStats(instances, classIndex)
        return leaves
    elif len(domains) == 1:
        if sameClass(instances, classIndex):
            total = len(table)
            return ['Leaves',[instances[0][classIndex],total,total,1.0]]
        else:
            partitions = partitionInstances(instances, atts[0], domains[0])
            
            subtree = [atts[0]]
            for key, value in partitions.items():
                valueList = [key]
                if len(value) < 1:
                    valueList.append(tdidt(instances, [], [], classIndex))
                else:
                    valueList.append(tdidt(value, [], [], classIndex))
                subtree.append(valueList)
            return subtree
    else:
        partitionAttIdx = selectAttributeRandom(instances, atts, classIndex, F)
        partitions = partitionInstances(instances, atts[partitionAttIdx], domains[partitionAttIdx])
        
        subtree = [atts[partitionAttIdx]]
        for key, value in partitions.items():
            valueList = [key]
            if len(value) < 1:
                valueList.append(tdidt(instances, [], [], classIndex))
            else:
                valueList.append(tdidt(value, 
                    [atts[i] for i in range(0,len(atts)) if i != partitionAttIdx], 
                    [domains[i] for i in range(0,len(domains)) if i != partitionAttIdx], classIndex))
            subtree.append(valueList)
        return subtree

# hw 6
# Random Forest
# Select next attribute from random subset        
def selectAttributeRandom(instances, attIndexes, classIndex, F):
    attIndexTemp = attIndexes[:]
    randIndexes = []
    if F >= len(attIndexes):
        randIndexes = attIndexes[:]
    else:
        shuffle(attIndexTemp)
        randIndexes = attIndexTemp[:F]
    Enews = [[calcEnew(instances,randIndex,classIndex),randIndex] for randIndex in randIndexes]
    sortedEnews = sorted(Enews, key=lambda x: x[0])
    return attIndexes.index(sortedEnews[0][1])

# hw 6
# Random Forest
# get accuracy for interim tree   
def determineAccuracy(tree,test, class_index, numClass, idxDict):
    confusionMat = [[0 for p in range(0,numClass+2)] for l in range(0,numClass)]
    
    for t in test:
        prediction, actual = tdidtClassifier(tree, t, class_index, 0)
        confusionMat[idxDict[actual]][idxDict[prediction]] += 1
            
    accuracy = calcAccuracy(confusionMat, numClass)
    
    return accuracy

# hw 6
# Random Forest
# get accuracy and cunfusion matrix for output
def confusionMatrix(class_idx, numClass, test, forest, idxDict, oppDict):
    confusionMat = [[0 for p in range(0,numClass+2)] for l in range(0,numClass)]
    
    for t in test:
        prediction, actual = randomForestClassifier(forest, t, class_idx, 0)
        confusionMat[idxDict[actual]][idxDict[prediction]] += 1
            
    accuracy = calcAccuracy(confusionMat, numClass)
    
    for w in range(0, numClass):
        trues = confusionMat[w][w]
        total = sum(confusionMat[w])
        confusionMat[w][numClass] = trues
        if total > 0:
            confusionMat[w][numClass+1] = trues / (total*1.0) * 100
        else:
            confusionMat[w][numClass+1] = 0
        trues = confusionMat[w][w]
        total = sum(confusionMat[w])
        confusionMat[w][numClass] = trues
        if total > 0:
            confusionMat[w][numClass+1] = trues / (total*1.0) * 100
        else:
            confusionMat[w][+1] = 0
    
    outputMatrix = [[0 for p in range(0,numClass+3)] for l in range(0,numClass)]
    for i in range(0, numClass):
        for k in range(0, numClass+3):
            if k == 0:
                outputMatrix[i][k] = oppDict[i]
            else:
                outputMatrix[i][k] = confusionMat[i][k-1]
    
    return (accuracy, outputMatrix)

# hw 6
# Random Forest
# get best trees from the N trees
def chooseMBest(forest, M):
    result = []
    for tree in forest:
        i = 0
        added = False
        if result == []:
            result.append(tree[0])
            added = True
        for r in result:
            if r[1] < tree[1]:
                result.insert(i,tree[0])
                added = True
                break
            i = i + 1
        if added == False:
            result.append(tree[0])
        if len(result) > M:
            result.pop()
    return result

# hw 6
# Random Forest
# classifier    
def randomForestClassifier(forest, instance, classIdx, isPrinting):
    classDict = {}
    for tree in forest:
        predictedClass = getClassification(tree, instance)
        if predictedClass not in classDict.keys():
            classDict[predictedClass] = 0
        classDict[predictedClass] += 1
    
    mostProbClass = ""
    highestProb = 0
    
    for key in classDict.keys():
        if classDict[key] > highestProb:
            mostProbClass = key
            highestProb = classDict[key]
    
    prediction = mostProbClass
        
    actual = instance[classIdx]
    
    if isPrinting:
        print("class: "+prediction+", actual: "+actual)
    return (prediction,actual)

# hw 6
# Get randomly stratified test and remainder sets
def randomStratify(table, class_idx):
    test = []
    remainder = []
    class_tables = {}
    
    for row in table:
        if row[class_idx] not in class_tables.keys():
            class_tables[row[class_idx]] = []
        class_tables[row[class_idx]].append(row)
        
    for k in class_tables.keys():
        shuffle(class_tables[k])
        
    for k in class_tables.keys():
        oneThird = int(len(class_tables[k]) / 3)
        test = test + class_tables[k][0:oneThird+1]
        remainder = remainder + class_tables[k][oneThird:]
    return (test, remainder)

###### END FUNCTIONS FROM PREVIOUS HOMEWORKS ######

# checks if a value is a float
def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False

# Ranking of salary amounts
def getSalaryRanking(val):
    if val >= 250000:
        return '1'
    elif val >= 150000:
        return '2'
    elif val >= 60000:
        return '3'
    elif val >= 32000:
        return '4'
    elif val >= 23000:
        return '5'
    else:
        return '6'

# Bin salaries for future use
def binSalaries(table, salary_idxs):
    newTable = []

    for row in table:
        newRow = row
        for idx in salary_idxs:
            if not newRow[idx] == "NA" and not newRow[idx] == 'NA':
                newRow[idx] = getSalaryRanking(float(newRow[idx]))
        newTable.append(newRow)

    return newTable
    
# Format confusion matrix
def getOutputMatrix(confusionMat, classes, oppDict):
    for w in range(0, len(classes)):
        trues = confusionMat[w][w]
        total = sum(confusionMat[w])
        confusionMat[w][len(classes)] = total
        if total > 0:
            confusionMat[w][len(classes)+1] = trues / (total*1.0) * 100
        else:
            confusionMat[w][len(classes)+1] = 0
    
    outputMatrix = [[0 for p in range(0,len(classes)+3)] for l in range(0,len(classes))]
    for i in range(0, len(classes)):
        for k in range(0, len(classes)+3):
            if k == 0:
                outputMatrix[i][k] = oppDict[i]
            else:
                outputMatrix[i][k] = confusionMat[i][k-1]
                
    return outputMatrix

def main():
    
    # Get dataset
    tableFull = csvToTableSkipTitle("Employee_Salaries_2015.csv")

    # Bin slaries
    tableFull = binSalaries(tableFull, [2, 3])
    # Use part of dataset because it takes a long time to run
    #table = tableFull[:1500]
    table = tableFull

    # Summary stats
    '''
    genderDict = {}
    deptDict = {}
    salaryDict = {}
    categoryDict = {}
    payDict = {}
    positionDict = {}
    
    for row in table:
        if row[1] not in genderDict.keys():
            genderDict[row[1]] = 1
        else:
            genderDict[row[1]] += 1
        if row[6] not in deptDict.keys():
            deptDict[row[6]] = 1
        else:
            deptDict[row[6]] += 1
        if row[2] not in salaryDict.keys():
            salaryDict[row[2]] = 1
        else:
            salaryDict[row[2]] += 1
        if row[8] not in categoryDict.keys():
            categoryDict[row[8]] = 1
        else:
            categoryDict[row[8]] += 1
        if row[3] not in payDict.keys():
            payDict[row[3]] = 1
        else:
            payDict[row[3]] += 1
        if row[9] not in positionDict.keys():
            positionDict[row[9]] = 1
        else:
            positionDict[row[9]] += 1

    print genderDict
    print deptDict
    print salaryDict
    print categoryDict
    print payDict
    print positionDict
    '''

    # Get attribute domains
    deptAcronymName_map = {}
    deptAcr_5 = []
    departments_6 = []
    assignments_8 = []
    positions_9 = []
    
    for row in table:
        if row[5] not in deptAcr_5:
            deptAcr_5.append(row[5])
        if row[6] not in departments_6:
            departments_6.append(row[6])
            deptAcronymName_map[row[5]] = row[6]
        if row[8] not in assignments_8:
            assignments_8.append(row[8])
        if row[9] not in positions_9:
            positions_9.append(row[9])

    #print deptAcr_5
    #print departments_6
    #print len(departments_6)
    #print assignments_8
    #print len(assignments_8)
    #print positions_9
    #print len(positions_9)
    
    # Make dataset deleting instances with missing values
    missingDelTable = [row for row in table if (row[2] != "NA" and row[3] != "NA")]
    
    print("===========================================")
    title = "Data Set:"
    title += "Employee Salaries - "
    title += "2015 dataset from Montgomery County of Maryland"
    print(title)
    print("===========================================")

    # ===== Visualizations ====================
    print("===========================================")
    print("Visualizations")
    print("===========================================")

    # Use original table to get totals for classes
    # (Classes don't have any missing values)

    # bar chart
    makeBarChart(getCol(table,1),
                 "gender_bar.pdf", "Gender Frequency", "Genders")
    print "Made gender_bar.pdf"
    makeBarChart(getCol(table,5), "dept_bar.pdf",
                 "Department Frequency", "Departments")
    print "Made dept_bar.pdf"

    # use table with instances with missing data deleted
    makeMultFreq(missingDelTable, "Employee Gender by Annual Salary",
                 2, 1, "Annual Salary", "Count", "gender_multFreq.pdf",True)
    print "Made gender_multFreq.pdf"
    makeMultFreq(missingDelTable, "Employee Department by Annual Salary",
                 2, 5, "Annual Salary", "Count", "dept_multFreq.pdf",False)
    print "Made dept_multFreq.pdf"
    makeMultFreq(missingDelTable, "Employee Gender by Department",
                 5, 1, "Department", "Count", "gender_dept_multFreq.pdf",
                 False)
    print "Made gender_dept_multFreq.pdf"
    makeMultFreq(missingDelTable, "Employee Gender by Employment Type",
                 8, 1, "Employment Type", "Count",
                 "gender_employmentType_multFreq.pdf",
                 False)
    print "Made gender_employmentType_multFreq.pdf"
    
    # ===== Predicting Gender ====================
    print ""
    print("===========================================")
    print("Class Label 1: Predicting Gender")
    print("===========================================")

    class_idx = 1
    gender_1 = ["M", "F"]
    attr = {2:"Annual Salary",
            3:"Gross Pay",
            6:"Department",
            8:"Assignment-category",
            9:"Position Title"}
    atts = [2, 3, 6, 8, 9]
    domains = [
        ['1', '2', '3', '4', '5', '6'],
        ['1', '2', '3', '4', '5', '6'],
        departments_6,
        assignments_8,
        positions_9]
    
    # Make dataset replacing missing values with averages
    replaceMissingWMeaningfulAvg(
        table, "replacedWMeaningful_EmployeeSalaries_gender.txt", attr, class_idx)
    meaningfulTableGender = csvToTable(
        "replacedWMeaningful_EmployeeSalaries_gender.txt")
    '''
    # Use dataset deleting instances with missing values (tableDeleting)
    
    # K-nn
    # Naive Bayes
    # Decision Tree
    # Random Forrest

    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Dataset with instances with missing values removed")
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(missingDelTable, 10, class_idx)

    # K-NN
    # accuracy info
    confusionMatrixKNN = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    for i in range(0, 10):
        print ("      Working on Stratified Sample: " + str(i))
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClasses(tableTrain, class_idx)
        classProbs, idxDict, oppDict = getClassProbabilities(tableTrain, class_idx, classList)

        # Decision Tree
        # Rules over entire dataset
        tree = tdidt(tableTrain, atts, domains, class_idx)
        
        for testRow in tableTest:
            # K-NN
            prediction, actual = fiveNearestNeighbors(tableTrain,testRow, atts,0, class_idx)
            confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
            
            # NAIVE BAYES
            prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
            confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

            # Descision Tree
            prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
            confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1

    accuracy = calcAccuracy(confusionMatrixKNN, len(gender_1))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, gender_1, oppDict)
                
    print("K-NN (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixNB, len(gender_1))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, gender_1, oppDict)
                
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixDT, len(gender_1))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, gender_1, oppDict)
                
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''

    '''
    # Use dataset replacing missing values with averages
    print ""
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Dataset with missing values replaced with meaningful averages")
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(meaningfulTableGender, 10, class_idx)

    # K-NN
    # accuracy info
    #confusionMatrixKNN = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # Decision Tree
    # accuracy info
    #confusionMatrixDT = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    for i in range(0, 10):
        print ("      Working on Stratified Sample: " + str(i))
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClasses(tableTrain, class_idx)
        classProbs, idxDict, oppDict = getClassProbabilities(tableTrain, class_idx, classList)

        # Decision Tree
        # Rules over entire dataset
        #tree = tdidt(tableTrain, atts, domains, class_idx)
        
        for testRow in tableTest:
            # K-NN
            #prediction, actual = fiveNearestNeighbors(tableTrain,testRow, atts,0, class_idx)
            #confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
            
            # NAIVE BAYES
            prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
            confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

            # Descision Tree
            #prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
            #confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1
    
    accuracy = calcAccuracy(confusionMatrixKNN, len(gender_1))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, gender_1, oppDict)
                
    print("K-NN (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixNB, len(gender_1))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, gender_1, oppDict)
                
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    
    accuracy = calcAccuracy(confusionMatrixDT, len(gender_1))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, gender_1, oppDict)
                
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''

    # Bootstrapping
    print("   Dataset with missing values replaced with meaningful averages")
    print("   Bootstrap")
    
    # N, M, F
    N = 60
    M = 7
    F = 2
    
    test, remainder = randomStratify(meaningfulTableGender,class_idx)

    # K-NN
    # accuracy info
    #confusionMatrixKNN = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # Decision Tree
    # accuracy info
    #confusionMatrixDT = [[0 for p in range(0,len(gender_1)+2)] for l in range(0,len(gender_1))]

    # NAIVE BAYES
    # Get all class probabilities
    classList, classTables = getUniqueClasses(remainder, class_idx)
    classProbs, idxDict, oppDict = getClassProbabilities(remainder, class_idx, classList)

    # Decision Tree
    # Rules over entire dataset
    #tree = tdidt(remainder, atts, domains, class_idx)
            
    for testRow in test:
        # K-NN
        #prediction, actual = fiveNearestNeighbors(remainder,testRow, atts,0, class_idx)
        #confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
                
        # NAIVE BAYES
        prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
        confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

        # Descision Tree
        #prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
        #confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1
    '''
    accuracy = calcAccuracy(confusionMatrixKNN, len(gender_1))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, gender_1, oppDict)
                
    print("K-NN (Bootstrap) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''
    accuracy = calcAccuracy(confusionMatrixNB, len(gender_1))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, gender_1, oppDict)
                
    print("Naive Bayes (Bootstrap) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''
    accuracy = calcAccuracy(confusionMatrixDT, len(gender_1))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, gender_1, oppDict)
                
    print("Descision Tree (Bootstrap) :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    
    #RANDOM FORESTS
    forests = createRandomForests(remainder, class_idx, N, F, 2, idxDict, atts, domains)
    finalForests = chooseMBest(forests, M)
    accuracy, outputMatrix = confusionMatrix(class_idx, 2, test, finalForests, idxDict, oppDict)

    print("Random Forest (Gender) : accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    print("Random Forest :")
    headerLst = []
    headerLst.append("Gender")
    for c in gender_1:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''
    # ===== Predicting Department ====================
    print ""
    print("===========================================")
    print("Class Label 2: Predicting Department")
    print("===========================================")

    class_idx = 6
    gender_1 = ["M", "F"]
    attr = {1:"Gender",
            2:"Annual Salary",
            3:"Gross Pay",
            8:"Assignment-category",
            9:"Position Title"}
    atts = [1, 2, 3, 8, 9]
    domains = [
        gender_1,
        ['1', '2', '3', '4', '5', '6'],
        ['1', '2', '3', '4', '5', '6'],
        assignments_8,
        positions_9]

    # departments_6
    
    # Make dataset replacing missing values with averages
    replaceMissingWMeaningfulAvg(table, "replacedWMeaningful_EmployeeSalaries_dept.txt", attr, class_idx)
    meaningfulTableDept = csvToTable("replacedWMeaningful_EmployeeSalaries_dept.txt")

    # Make dictionaries for department indices
    departments_6
    idxDict = {}
    oppDict = {}
    for i in range(0, len(departments_6)):
        idxDict[departments_6[i]] = i
        oppDict[i] = deptAcr_5[i]
    '''
    # Use dataset deleting instances with missing values (tableDeleting)
    
    # K-nn
    # Naive Bayes
    # Decision Tree
    # Random Forrest

    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Dataset with instances with missing values removed")
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(missingDelTable, 10, class_idx)

    # K-NN
    # accuracy info
    confusionMatrixKNN = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # NAIVE BAYES
    # accuracy info
    confusionMatrixNB = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    for i in range(0, 10):
        print ("      Working on Stratified Sample: " + str(i))
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # NAIVE BAYES
        # Get all class probabilities
        classList, classTables = getUniqueClasses(tableTrain, class_idx)
        classProbs, _, _ = getClassProbabilities(tableTrain, class_idx, classList)
        
        # Decision Tree
        # Rules over entire dataset
        tree = tdidt(tableTrain, atts, domains, class_idx)
        
        for testRow in tableTest:
            # K-NN
            prediction, actual = fiveNearestNeighbors(tableTrain,testRow, atts,0, class_idx)
            confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
            
            # NAIVE BAYES
            prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
            confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

            # Descision Tree
            prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
            confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1

    accuracy = calcAccuracy(confusionMatrixKNN, len(departments_6))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, deptAcr_5, oppDict)
                
    print("K-NN (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixNB, len(departments_6))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, deptAcr_5, oppDict)
                
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixDT, len(departments_6))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, deptAcr_5, oppDict)
                
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    
    # Use dataset replacing missing values with averages
    print ""
    
    # STRATIFIED K-FOLD CROSS VALIDATION
    print("   Dataset with missing values replaced with meaningful averages")
    print("   Stratified 10-fold Cross Validation")

    stratifiedResults = stratifiedSubsamplesGen(meaningfulTableDept, 10, class_idx)

    # K-NN
    # accuracy info
    #confusionMatrixKNN = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # NAIVE BAYES
    # accuracy info
    #confusionMatrixNB = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    for i in range(0, 10):
        print ("      Working on Stratified Sample: " + str(i))
        tableTrain = []
        for k in range(0,10):
            if k != i:
                tableTrain += stratifiedResults[k]
        tableTest = stratifiedResults[i]

        # NAIVE BAYES
        # Get all class probabilities
        #classList, classTables = getUniqueClasses(tableTrain, class_idx)
        #classProbs, _, _ = getClassProbabilities(tableTrain, class_idx, classList)

        # Decision Tree
        # Rules over entire dataset
        tree = tdidt(tableTrain, atts, domains, class_idx)
        
        for testRow in tableTest:
            # K-NN
            #prediction, actual = fiveNearestNeighbors(tableTrain,testRow, atts,0, class_idx)
            #confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
            
            # NAIVE BAYES
            #prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
            #confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

            # Descision Tree
            prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
            confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1
    
    accuracy = calcAccuracy(confusionMatrixKNN, len(departments_6))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, deptAcr_5, oppDict)
                
    print("K-NN (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixNB, len(departments_6))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, deptAcr_5, oppDict)
                
    print("Naive Bayes (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    
    accuracy = calcAccuracy(confusionMatrixDT, len(departments_6))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, deptAcr_5, oppDict)
    
    print("Descision Tree (Stratified 10-Fold cross Validation Results) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''

    # Bootstrapping
    print("   Dataset with missing values replaced with meaningful averages")
    print("   Bootstrap")
    
    # N, M, F
    N = 60
    M = 7
    F = 2
    
    test, remainder = randomStratify(meaningfulTableDept,class_idx)

    # K-NN
    # accuracy info
    #confusionMatrixKNN = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # NAIVE BAYES
    # accuracy info
    #confusionMatrixNB = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # Decision Tree
    # accuracy info
    confusionMatrixDT = [[0 for p in range(0,len(departments_6)+2)] for l in range(0,len(departments_6))]

    # NAIVE BAYES
    # Get all class probabilities
    #classList, classTables = getUniqueClasses(remainder, class_idx)
    #classProbs, _, _ = getClassProbabilities(remainder, class_idx, classList)

    # Decision Tree
    # Rules over entire dataset
    tree = tdidt(remainder, atts, domains, class_idx)
        
    for testRow in test:
        # K-NN
        #prediction, actual = fiveNearestNeighbors(remainder,testRow, atts,0, class_idx)
        #confusionMatrixKNN[idxDict[actual]][idxDict[prediction]] += 1
            
        # NAIVE BAYES
        #prediction, actual = naiveBayesComparison(testRow, class_idx, classProbs, classTables, atts, 0)
        #confusionMatrixNB[idxDict[actual]][idxDict[prediction]] += 1

        # Descision Tree
        prediction, actual = tdidtClassifier(tree, testRow, class_idx, 0)
        confusionMatrixDT[idxDict[actual]][idxDict[prediction]] += 1
    '''
    accuracy = calcAccuracy(confusionMatrixKNN, len(departments_6))
    print("      k Nearest Neighbors: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixKNN, deptAcr_5, oppDict)
                
    print("K-NN (Bootstrap) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")

    accuracy = calcAccuracy(confusionMatrixNB, len(departments_6))
    print("      Naive Bayes: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    
    outputMatrix = getOutputMatrix(confusionMatrixNB, deptAcr_5, oppDict)
                
    print("Naive Bayes (Bootstrap) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''
    accuracy = calcAccuracy(confusionMatrixDT, len(departments_6))
    print("      Descision Tree: accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
                
    outputMatrix = getOutputMatrix(confusionMatrixDT, deptAcr_5, oppDict)
    
    print("Descision Tree (Bootstrap) :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    
    # Print tree rules
    #printRules(tree)
    print("")
    '''
    #RANDOM FORESTS
    forests = createRandomForests(remainder, class_idx, N, F, len(deptAcr_5), idxDict, atts, domains)
    finalForests = chooseMBest(forests, M)
    accuracy, outputMatrix = confusionMatrix(class_idx, len(deptAcr_5), test, finalForests, idxDict, oppDict)

    print("Random Forest (Department) : accuracy: " + str(accuracy) + ", error rate: " + str(calcErrorRate(accuracy)))
    print ""
    print("Random Forest :")
    headerLst = []
    headerLst.append("Department")
    for c in deptAcr_5:
        headerLst.append(c)
    headerLst.append("Total")
    headerLst.append("Recognition (%)")
    tableView = tabulate.tabulate(outputMatrix,headers=headerLst,tablefmt="rst")
    print(tableView)
    print("")
    '''
    print ("Map from acronym to department")
    for dept in deptAcr_5:
        print (dept + " = " + deptAcronymName_map[dept])

if __name__ == '__main__':
    main()
