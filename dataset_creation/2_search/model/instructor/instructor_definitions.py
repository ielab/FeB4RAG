

# Code taken from https://github.com/xlang-ai/instructor-embedding/blob/e69a1a51482c2759795e2ed3f76fc620d8e7e13d/evaluation/MTEB/mteb/abstasks/AbsTaskRetrieval.py
# Additional instructions are added for the following datasets from BEIR: signal1m, robust04, trec-news

DEFINITIONS_INSTRUCTOR = {
    'hkunlp/instructor-xl': {
        'signal1m':
            {
                'query': 'Represent the news article title for retrieving relevant tweets: ',
                'corpus': 'Represent the tweet for retrieval: ',
            },
        'robust04':
            {
                'query': 'Represent the news query for retrieving supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'trec-news':
            {
                'query': 'Represent the news headline for retrieving  supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'hotpotqa':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'msmarco':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'dbpedia-entity':
            {
                'query': 'Represent the Wikipedia questions to retrieve a supporting document: ',
                'corpus': 'Represent the Wikipedia documents for retrieval: ',
            },
        'nq':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'quora':
            {
                'query': 'Represent the Quora question to retrieve question: ',
                'corpus': 'Represent the Quora question to retrieve question: ',
            },
        'scidocs':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'trec-covid':
            {
                'query': 'Represent the Coronavirus questions to retrieve a supporting document: ',
                'corpus': 'Represent the Coronavirus documents for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent questions: ',
                'corpus': 'Represent arguments: ',
            },
        'scifact':
            {
                'query': 'Represent the Scientific queries for retrieving a supporting passage: ',
                'corpus': 'represent the scientific paragraph for retrieval: ',
            },
        'nfcorpus':
            {
                'query': 'Represent the nutrition facts to retrieve Public medical articles: ',
                'corpus': 'Represent the Public medical articles for retrieval: ',
            },
        'arguana':
            {
                'query': 'Represent Debating conversations to retrieve a counter-argument: ',
                'corpus': 'Represent counter-arguments: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix questions to retrieve a supporting answer: ',
                'corpus': 'Represent the Unix answers for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'fiqa':
            {
                'query': 'Represent the finance questions to retrieve a supporting answer: ',
                'corpus': 'Represent the finance answers for retrieval: ',
            },
    },
    'hkunlp/instructor-large':{
        'signal1m':
            {
                'query': 'Represent the news article title for retrieving relevant tweets: ',
                'corpus': 'Represent the tweet for retrieval: ',
            },
        'robust04':
            {
                'query': 'Represent the news query for retrieving supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'trec-news':
            {
                'query': 'Represent the news headline for retrieving  supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'hotpotqa':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'msmarco':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'dbpedia-entity':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'nq':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'quora':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'scidocs':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'trec-covid':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'scifact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'nfcorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'arguana':
            {
                'query': 'Represent a Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent a Counter-argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'fiqa':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
    },
    'hkunlp/instructor-base': {
        'signal1m':
            {
                'query': 'Represent the news article title for retrieving relevant tweets: ',
                'corpus': 'Represent the tweet for retrieval: ',
            },
        'robust04':
            {
                'query': 'Represent the news query for retrieving supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'trec-news':
            {
                'query': 'Represent the news headline for retrieving  supporting news articles: ',
                'corpus': 'Represent the news article for retrieval: ',
            },
        'ClimateFEVER':
            {
                'query': 'Represent the Climate question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'hotpotqa':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'FEVER':
            {
                'query': 'Represent the fact for retrieving supporting evidence: ',
                'corpus': 'Represent the evidence for retrieval: ',
            },
        'msmarco':
            {
                'query': 'Represent the question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'dbpedia-entity':
            {
                'query': 'Represent the Wikipedia sentence for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'nq':
            {
                'query': 'Represent the Wikipedia question for retrieving supporting documents: ',
                'corpus': 'Represent the document for retrieval: ',
            },
        'quora':
            {
                'query': 'Represent the Quora question for retrieving duplicate questions: ',
                'corpus': 'Represent the Quora question for retrieving duplicate questions: ',
            },
        'scidocs':
            {
                'query': 'Represent a Science question for retrieving supporting papers: ',
                'corpus': 'Represent the Science paper: ',
            },
        'trec-covid':
            {
                'query': 'Represent the Coronavirus question for retrieving supporting documents: ',
                'corpus': 'Represent the Coronavirus document for retrieval: ',
            },
        'Touche2020':
            {
                'query': 'Represent a question: ',
                'corpus': 'Represent an argument: ',
            },
        'scifact':
            {
                'query': 'Represent a Scientific query for retrieving a supporting passage; ',
                'corpus': 'represent the Scientific passage for retrieval; ',
            },
        'nfcorpus':
            {
                'query': 'Represent the Medicine question for retrieving a relevant document: ',
                'corpus': 'Represent the medical document for retrieval: ',
            },
        'arguana':
            {
                'query': 'Represent the Debate argument for retrieving a counter-argument: ',
                'corpus': 'Represent the Counter debate argument: ',
            },
        'CQADupstackTexRetrieval':
            {
                'query': 'Represent the question for retrieving answers: ',
                'corpus': 'Represent the answer for retrieval: ',
            },
        'CQADupstackWebmastersRetrieval':
            {
                'query': 'Represent the Webmaster question for retrieving answers: ',
                'corpus': 'Represent the Webmaster answer: ',
            },
        'CQADupstackEnglishRetrieval':
            {
                'query': 'Represent the English question for retrieving documents: ',
                'corpus': 'Represent the English answer for retrieval: ',
            },
        'CQADupstackGamingRetrieval':
            {
                'query': 'Represent the Gaming question for retrieving answers: ',
                'corpus': 'Represent the Gaming answer for retrieval: ',
            },
        'CQADupstackGisRetrieval':
            {
                'query': 'Represent the Gis question for retrieving answers: ',
                'corpus': 'Represent the Gis answer for retrieval: ',
            },
        'CQADupstackUnixRetrieval':
            {
                'query': 'Represent the Unix question for retrieving answers: ',
                'corpus': 'Represent the Unix answer for retrieval: ',
            },
        'CQADupstackMathematicaRetrieval':
            {
                'query': 'Represent the Mathematical question for retrieving answers: ',
                'corpus': 'Represent the Mathematical answer for retrieval: ',
            },
        'CQADupstackStatsRetrieval':
            {
                'query': 'Represent the Statistical question for retrieving answers: ',
                'corpus': 'Represent the Statistical answer for retrieval: ',
            },
        'CQADupstackPhysicsRetrieval':
            {
                'query': 'Represent the Physics question for retrieving answers: ',
                'corpus': 'Represent the Physics answer for retrieval: ',
            },
        'CQADupstackProgrammersRetrieval':
            {
                'query': 'Represent the Programming question for retrieving answers: ',
                'corpus': 'Represent the Programming answer for retrieval: ',
            },
        'CQADupstackAndroidRetrieval':
            {
                'query': 'Represent the Android question for retrieving answers: ',
                'corpus': 'Represent the Android answer for retrieval: ',
            },
        'CQADupstackWordpressRetrieval':
            {
                'query': 'Represent the Wordpress question for retrieving answers: ',
                'corpus': 'Represent the Wordpress answer for retrieval: ',
            },
        'fiqa':
            {
                'query': 'Represent the finance question for retrieving the supporting answers: ',
                'corpus': 'Represent the finance answer for retrieval: ',
            },
    },
}

