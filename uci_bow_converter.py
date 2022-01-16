

def write_vocab_csv(l):
  with open('/home/vt/extra_storage/Production/output/vocab.txt', 'w') as f:
        for elem in l:
            f.write(elem + "\n")

def custom_stop_words():
    stopwordList = ["rt"] 
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optional
    return(stopwordList)
    
input_tweets_folder = "text.json"
data = spark.read.option("header", "false").option("multiline",False).json(input_tweets_folder)
data = data.na.drop()
data = data.withColumn('text_cleaned', regexp_replace('text', '[#|,|&|!|~|*]|http.*', ''))
# ------------------- Grouping individual texts ---------------------- #
grouped_df = data.groupby('name').agg(collect_list('text_cleaned').alias("text_aggregated"))
grouped_appended_df = grouped_df.withColumn("text_aggregated_1", concat_ws(". ", "text_aggregated"))
# --------------------------------------------------------------------#
tokenizer = Tokenizer(inputCol="text_aggregated_1", outputCol="words")
wordsData = tokenizer.transform(grouped_appended_df)
stopwordlist = custom_stop_words()
remover = StopWordsRemover(inputCol="words", outputCol="filtered_col", stopWords=stopwordlist)
filtered = remover.transform(wordsData)
filtered.select("filtered_col").show(truncate=False)
cv = CountVectorizer(inputCol="filtered_col", outputCol="rawFeatures", vocabSize=15000, minDF=5.0)
cvmodel = cv.fit(filtered)
# ------------------------- BOW model ----------------#
vocab = cvmodel.vocabulary # list of words in order, write out for vocab.txt file
d_bow = filtered.count()   # number of documents in the vocabulary
w_bow = len(vocab)         # number of words in the vocabulary
vector_udf = udf(lambda vector: vector.numNonzeros(), LongType())
# ------------------------ Write out docword.txt------------------------#
vocab_broadcast = sc.broadcast(vocab)
featurized = cvmodel.transform(filtered)
nnz_bow = featurized.select(vector_udf('rawFeatures')).groupBy().sum().collect()
sparse_values = udf(lambda v: v.values.tolist(), ArrayType(DoubleType()))
nnz_elements_count = featurized.select(sparse_values('rawFeatures'))
sparse_indices = udf(lambda v: v.indices.tolist(), ArrayType(LongType()))
nnz_elements = featurized.select(sparse_indices('rawFeatures'))
fzipped = f.select('vals','indices').rdd.zipWithIndex().toDF()
fzipped_sep = fzipped.withColumn('vals', fzipped['_1'].getItem("vals"))
fzipped_sep = fzipped_sep.withColumn('indices', fzipped['_1'].getItem("indices"))
fzipped_sep2 = fzipped_sep.select("_2", arrays_zip("indices", "vals"))
nnz_data = fzipped_sep2.select("_2", explode("arrays_zip(indices, vals)"))
out2 = nnz_data.withColumn('indices', out['col'].getItem("indices")).withColumn('cnt', out['col'].getItem("vals")).withColumn('reindexed', out2['_2'] + 1).select('reindexed', 'indices', col('cnt').cast(IntegerType()))
nnz_bow = out2.select(sum(col('cnt'))).collect()[0][0] # number of nnz in the documents
# ------------------------- Write docword.txt file ------------------------#
out2.repartition(1).write.save(path='docword.txt', format='csv', mode='overwrite', sep=" ")
# ------------------------- Write the vocab file --------------------------#
write_vocab_csv(vocab)

