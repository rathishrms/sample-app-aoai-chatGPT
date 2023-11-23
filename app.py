import json
import os
import logging
import requests
import openai
import copy
import utils
import uuid
import time
import re
import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from flask import Flask, Response, request, jsonify, send_from_directory
from dotenv import load_dotenv
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table

from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient

load_dotenv()

app = Flask(__name__, static_folder="static")

# Static Files
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file('favicon.ico')

@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("static/assets", path)

# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
DEBUG_LOGGING = DEBUG.lower() == "true"
if DEBUG_LOGGING:
    logging.basicConfig(level=logging.DEBUG)

# On Your Data Settings
DATASOURCE_TYPE = os.environ.get("DATASOURCE_TYPE", "AzureCognitiveSearch")
SEARCH_TOP_K = os.environ.get("SEARCH_TOP_K", 5)
SEARCH_STRICTNESS = os.environ.get("SEARCH_STRICTNESS", 3)
SEARCH_ENABLE_IN_DOMAIN = os.environ.get("SEARCH_ENABLE_IN_DOMAIN", "true")

# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get("AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get("AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", SEARCH_TOP_K)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get("AZURE_SEARCH_ENABLE_IN_DOMAIN", SEARCH_ENABLE_IN_DOMAIN)
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")
AZURE_SEARCH_VECTOR_COLUMNS = os.environ.get("AZURE_SEARCH_VECTOR_COLUMNS")
AZURE_SEARCH_QUERY_TYPE = os.environ.get("AZURE_SEARCH_QUERY_TYPE")
AZURE_SEARCH_PERMITTED_GROUPS_COLUMN = os.environ.get("AZURE_SEARCH_PERMITTED_GROUPS_COLUMN")
AZURE_SEARCH_STRICTNESS = os.environ.get("AZURE_SEARCH_STRICTNESS", SEARCH_STRICTNESS)

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get("AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get("AZURE_OPENAI_PREVIEW_API_VERSION", "2023-08-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo-16k") # Name of the model, e.g. 'gpt-35-turbo-16k' or 'gpt-4'
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME", "")
OPENAI_DEPLOYMENT_NAME = os.environ.get("OPENAI_DEPLOYMENT_NAME")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
OPENAI_DEPLOYMENT_ENDPOINT = os.environ.get("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")
OPENAI_DEPLOYMENT_VERSION = os.environ.get("OPENAI_DEPLOYMENT_VERSION")
AAD_TENANT_ID = os.environ.get("AAD_TENANT_ID")
KUSTO_CLUSTER = os.environ.get("KUSTO_CLUSTER")
KUSTO_DATABASE = os.environ.get("KUSTO_DATABASE")
KUSTO_TABLE = os.environ.get("KUSTO_TABLE")
KUSTO_MANAGED_IDENTITY_APP_ID = os.environ.get("KUSTO_MANAGED_IDENTITY_APP_ID")
KUSTO_MANAGED_IDENTITY_SECRET = os.environ.get("KUSTO_MANAGED_IDENTITY_SECRET")

# CosmosDB Mongo vcore vector db Settings
AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING")  #This has to be secure string
AZURE_COSMOSDB_MONGO_VCORE_DATABASE = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_DATABASE")
AZURE_COSMOSDB_MONGO_VCORE_CONTAINER = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTAINER")
AZURE_COSMOSDB_MONGO_VCORE_INDEX = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_INDEX")
AZURE_COSMOSDB_MONGO_VCORE_TOP_K = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TOP_K", AZURE_SEARCH_TOP_K)
AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS", AZURE_SEARCH_STRICTNESS)  
AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN", AZURE_SEARCH_ENABLE_IN_DOMAIN)
AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS", "")
AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN")
AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS = os.environ.get("AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS")

print(AZURE_OPENAI_KEY)

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False

# Chat History CosmosDB Integration Settings
AZURE_COSMOSDB_DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE")
AZURE_COSMOSDB_ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT")
AZURE_COSMOSDB_CONVERSATIONS_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER")
AZURE_COSMOSDB_ACCOUNT_KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY")

# Connect to adx using AAD app registration
cluster = KUSTO_CLUSTER
kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, KUSTO_MANAGED_IDENTITY_APP_ID, KUSTO_MANAGED_IDENTITY_SECRET,  AAD_TENANT_ID)
client = KustoClient(kcsb)

template_prefix = """
<|im_start|>system
I have an Azure Data Explorer (Kusto) table containing the following columns: 
eventTimeFlow,flowRecord_flowRecordType,flowRecord_subscriberInfo_imsi,flowRecord_keys_sessionId,flowRecord_dpiStringInfo_application,flowRecord_dpiStringInfo_layer7Protocol,flowRecord_gatewayInfo_gwNodeID,flowRecord_dpiStringInfo_operatingSystem,flowRecord_creationtime_timesecs,flowRecord_creationtime_timeusecs,flowRecord_networkStatsInfo_downlinkFlowPeakThroughput,flowRecord_networkStatsInfo_uplinkFlowPeakThroughput,flowRecord_networkStatsInfo_downlinkFlowActivityDuration,flowRecord_networkStatsInfo_uplinkFlowActivityDuration,flowRecord_networkStatsInfo_downlinkInitialRTT_timesecs,flowRecord_networkStatsInfo_downlinkInitialRTT_timeusecs,flowRecord_networkStatsInfo_uplinkInitialRTT_timesecs,flowRecord_networkStatsInfo_uplinkInitialRTT_timeusecs,flowRecord_networkStatsInfo_closureReason,flowRecord_networkPerfInfo_initialRTT_timesecs,flowRecord_networkPerfInfo_initialRTT_timeusecs,flowRecord_networkPerfInfo_HttpTtfbTime_timesecs,flowRecord_networkPerfInfo_HttpTtfbTime_timeusecs,flowRecord_dataStats_upLinkOctets,flowRecord_dataStats_downLinkOctets,flowRecord_dataStats_downLinkPackets,flowRecord_dataStats_downLinkDropPackets,flowRecord_dataStats_upLinkPackets,flowRecord_dataStats_upLinkDropPackets,flowRecord_tcpRetransInfo_downlinkRetransBytes,flowRecord_tcpRetransInfo_uplinkRetransBytes,flowRecord_tcpRetransInfo_downlinkRetransPackets,flowRecord_tcpRetransInfo_uplinkRetransPackets,flowRecord_ipTuple_networkIpAddress,flowRecord_ipTuple_networkPort,flowRecord_ipTuple_protocol,flowRecord_keys_flowId,eventTimeSession,sessionRecord_subscriberInfo_userLocationInfo_ecgi_eci,sessionRecord_keys_sessionId,sessionRecord_subscriberInfo_imsi,sessionRecord_subscriberInfo_msisdn,sessionRecord_subscriberInfo_imeisv_tac,sessionRecord_servingNetworkInfo_apnId,sessionRecord_servingNetworkInfo_nodeAddress,sessionRecord_servingNetworkInfo_nodePlmnId_mcc,sessionRecord_servingNetworkInfo_nodePlmnId_mnc,eventTimeHTTP,httpRecord_keys_flowId,httpRecord_keys_sessionId,httpRecord_httpTransactionRecord_subscriberInformation_gwNodeID,httpRecord_httpTransactionRecord_subscriberInformation_imsi,httpRecord_keys_transactionId,httpRecord_httpTransactionRecord_dpiInformation_application,httpRecord_httpTransactionRecord_subscriberInformation_realApn,httpRecord_httpTransactionRecord_requestInformation_failureReason,httpRecord_httpTransactionRecord_responseInformation_responseStatus,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_maxDownstreamMSS,httpRecord_httpTransactionRecord_tcpServerConnectionInformation_MSS,httpRecord_httpTransactionRecord_tcpServerConnectionInformation_rtt,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelNoneTime,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelMildTime,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelModerateTime,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelSevereTime,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_minRTT,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_maxRTT,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_avgRTT,httpRecord_httpTransactionRecord_tcpSplicingInformation_tlsSNI,httpRecord_httpTransactionRecord_udpProxyInformation_quicSni,httpRecord_httpTransactionRecord_requestInformation_serverUrl_Host,Column71


Write an KQL query based on the user input below. Answer in a concise manner. Answer only with the KQL query where the table name is T, no extra text.

user input: 
"""
template_sufix = "<|im_end|>\n<|im_start|>assistant"

# Define functions to call the OpenAI API and run KQL query
def call_openai_new(messages):
    response = openai.ChatCompletion.create(
        engine=utils.OPENAI_DEPLOYMENT_NAME,
        messages = messages,
        temperature=float(AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
        top_p=float(AZURE_OPENAI_TOP_P),
        stop=AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        stream=SHOULD_STREAM
    )
    return response

def call_openai_kql_response_new(messages):
    response = call_openai_new(messages)

    # print("Raw response:", response)
    # query = response.replace("T", "['enriched-edr']")
    # # query = response.replace("T", "monitor")
    # query = query.replace("```", "")
    # # print("Generated KQL query:", query)
    # response = client.execute("aoienriched", query)

    df = dataframe_from_result_table(response.primary_results[0])
    return df

def call_openai(template_prefix, text):
    prompt = template_prefix + text + template_sufix
    response = openai.Completion.create(
        model=utils.OPENAI_DEPLOYMENT_NAME,
        engine=utils.OPENAI_DEPLOYMENT_NAME,
        prompt=prompt,
        temperature=0,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["<|im_end|>"])
    # print(response)
    # response = response['choices'][0]['text']
    response = response.choices[0].text.strip()
    response = utils.remove_chars("\n", response)
    response=utils.start_after_string("Answer:", response)
    response=utils.remove_tail_tags("<|im_end|>", response)
    return response

def call_openai_kql_response(text):
    response = call_openai(template_prefix, text)

    # print("Raw response:", response)
    query = response.replace("T", "['enriched-edr']")
    # query = response.replace("T", "monitor")
    query = query.replace("```", "")
    # print("Generated KQL query:", query)
    response = client.execute("aoienriched", query)

    df = dataframe_from_result_table(response.primary_results[0])
    return df

# df = call_openai_kql_response("calculate the total uplink and downlink octets for each application in the 'enriched-edr' table.")
# print(df)

# df = call_openai_kql_response("the distribution of sessions across different applications:")
# print(df)

df = call_openai_kql_response("Total uplink and downlink octets by application")
print(df)

# Initialize a CosmosDB client with AAD auth and containers for Chat History
cosmos_conversation_client = None
if AZURE_COSMOSDB_DATABASE and AZURE_COSMOSDB_ACCOUNT and AZURE_COSMOSDB_CONVERSATIONS_CONTAINER:
    try :
        cosmos_endpoint = f'https://{AZURE_COSMOSDB_ACCOUNT}.documents.azure.com:443/'

        if not AZURE_COSMOSDB_ACCOUNT_KEY:
            credential = DefaultAzureCredential()
        else:
            credential = AZURE_COSMOSDB_ACCOUNT_KEY

        cosmos_conversation_client = CosmosConversationClient(
            cosmosdb_endpoint=cosmos_endpoint, 
            credential=credential, 
            database_name=AZURE_COSMOSDB_DATABASE,
            container_name=AZURE_COSMOSDB_CONVERSATIONS_CONTAINER
        )
    except Exception as e:
        logging.exception("Exception in CosmosDB initialization", e)
        cosmos_conversation_client = None


def is_chat_model():
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k', 'gpt-35-turbo-16k']:
        return True
    return False

def should_use_data():
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY:
        if DEBUG_LOGGING:
            logging.debug("Using Azure Cognitive Search")
        return True
    
    if AZURE_COSMOSDB_MONGO_VCORE_DATABASE and AZURE_COSMOSDB_MONGO_VCORE_CONTAINER and AZURE_COSMOSDB_MONGO_VCORE_INDEX and AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING:
        if DEBUG_LOGGING:
            logging.debug("Using Azure CosmosDB Mongo vcore")
        return True
    
    return False


def format_as_ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"

def fetchUserGroups(userToken, nextLink=None):
    # Recursively fetch group membership
    if nextLink:
        endpoint = nextLink
    else:
        endpoint = "https://graph.microsoft.com/v1.0/me/transitiveMemberOf?$select=id"
    
    headers = {
        'Authorization': "bearer " + userToken
    }
    try :
        r = requests.get(endpoint, headers=headers)
        if r.status_code != 200:
            if DEBUG_LOGGING:
                logging.error(f"Error fetching user groups: {r.status_code} {r.text}")
            return []
        
        r = r.json()
        if "@odata.nextLink" in r:
            nextLinkData = fetchUserGroups(userToken, r["@odata.nextLink"])
            r['value'].extend(nextLinkData)
        
        return r['value']
    except Exception as e:
        logging.error(f"Exception in fetchUserGroups: {e}")
        return []


def generateFilterString(userToken):
    # Get list of groups user is a member of
    userGroups = fetchUserGroups(userToken)

    # Construct filter string
    if not userGroups:
        logging.debug("No user groups found")

    group_ids = ", ".join([obj['id'] for obj in userGroups])
    return f"{AZURE_SEARCH_PERMITTED_GROUPS_COLUMN}/any(g:search.in(g, '{group_ids}'))"



def prepare_body_headers_with_data(request):
    request_messages = request.json["messages"]

    body = {
        "messages": request_messages,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "dataSources": []
    }

    if DATASOURCE_TYPE == "AzureCognitiveSearch":
        # Set query type
        query_type = "simple"
        if AZURE_SEARCH_QUERY_TYPE:
            query_type = AZURE_SEARCH_QUERY_TYPE
        elif AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG:
            query_type = "semantic"

        # Set filter
        filter = None
        userToken = None
        if AZURE_SEARCH_PERMITTED_GROUPS_COLUMN:
            userToken = request.headers.get('X-MS-TOKEN-AAD-ACCESS-TOKEN', "")
            if DEBUG_LOGGING:
                logging.debug(f"USER TOKEN is {'present' if userToken else 'not present'}")

            filter = generateFilterString(userToken)
            if DEBUG_LOGGING:
                logging.debug(f"FILTER: {filter}")

        body["dataSources"].append(
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentFields": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None,
                        "vectorFields": AZURE_SEARCH_VECTOR_COLUMNS.split("|") if AZURE_SEARCH_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_SEARCH_TOP_K,
                    "queryType": query_type,
                    "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE,
                    "filter": filter,
                    "strictness": int(AZURE_SEARCH_STRICTNESS)
                }
            })
    elif DATASOURCE_TYPE == "AzureCosmosDB":
        # Set query type
        query_type = "vector"

        body["dataSources"].append(
            {
                "type": "AzureCosmosDB",
                "parameters": {
                    "connectionString": AZURE_COSMOSDB_MONGO_VCORE_CONNECTION_STRING,
                    "indexName": AZURE_COSMOSDB_MONGO_VCORE_INDEX,
                    "databaseName": AZURE_COSMOSDB_MONGO_VCORE_DATABASE,
                    "containerName": AZURE_COSMOSDB_MONGO_VCORE_CONTAINER,                    
                    "fieldsMapping": {
                        "contentFields": AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS.split("|") if AZURE_COSMOSDB_MONGO_VCORE_CONTENT_COLUMNS else [],
                        "titleField": AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_TITLE_COLUMN else None,
                        "urlField": AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_URL_COLUMN else None,
                        "filepathField": AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN if AZURE_COSMOSDB_MONGO_VCORE_FILENAME_COLUMN else None,
                        "vectorFields": AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS.split("|") if AZURE_COSMOSDB_MONGO_VCORE_VECTOR_COLUMNS else []
                    },
                    "inScope": True if AZURE_COSMOSDB_MONGO_VCORE_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_COSMOSDB_MONGO_VCORE_TOP_K,
                    "strictness": int(AZURE_COSMOSDB_MONGO_VCORE_STRICTNESS),
                    "queryType": query_type,
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            })
    else:
        raise Exception(f"DATASOURCE_TYPE is not configured or unknown: {DATASOURCE_TYPE}")

    if "vector" in query_type.lower():
        if AZURE_OPENAI_EMBEDDING_NAME:
            body["dataSources"][0]["parameters"]["embeddingDeploymentName"] = AZURE_OPENAI_EMBEDDING_NAME
        else:
            body["dataSources"][0]["parameters"]["embeddingEndpoint"] = AZURE_OPENAI_EMBEDDING_ENDPOINT
            body["dataSources"][0]["parameters"]["embeddingKey"] = AZURE_OPENAI_EMBEDDING_KEY

    if DEBUG_LOGGING:
        body_clean = copy.deepcopy(body)
        if body_clean["dataSources"][0]["parameters"].get("key"):
            body_clean["dataSources"][0]["parameters"]["key"] = "*****"
        if body_clean["dataSources"][0]["parameters"].get("connectionString"):
            body_clean["dataSources"][0]["parameters"]["connectionString"] = "*****"
        if body_clean["dataSources"][0]["parameters"].get("embeddingKey"):
            body_clean["dataSources"][0]["parameters"]["embeddingKey"] = "*****"
            
        logging.debug(f"REQUEST BODY: {json.dumps(body_clean, indent=4)}")

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_KEY,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/3.0.0"
    }

    return body, headers


def stream_with_data(body, headers, endpoint, history_metadata={}):
    s = requests.Session()
    try:
        with s.post(endpoint, json=body, headers=headers, stream=True) as r:
            for line in r.iter_lines(chunk_size=10):
                response = {
                    "id": "",
                    "model": "",
                    "created": 0,
                    "object": "",
                    "choices": [{
                        "messages": []
                    }],
                    "apim-request-id": "",
                    'history_metadata': history_metadata
                }
                if line:
                    if AZURE_OPENAI_PREVIEW_API_VERSION == '2023-06-01-preview':
                        lineJson = json.loads(line.lstrip(b'data:').decode('utf-8'))
                    else:
                        try:
                            rawResponse = json.loads(line.lstrip(b'data:').decode('utf-8'))
                            lineJson = formatApiResponseStreaming(rawResponse)
                        except json.decoder.JSONDecodeError:
                            continue

                    if 'error' in lineJson:
                        yield format_as_ndjson(lineJson)
                    response["id"] = lineJson["id"]
                    response["model"] = lineJson["model"]
                    response["created"] = lineJson["created"]
                    response["object"] = lineJson["object"]
                    response["apim-request-id"] = r.headers.get('apim-request-id')

                    role = lineJson["choices"][0]["messages"][0]["delta"].get("role")

                    if role == "tool":
                        response["choices"][0]["messages"].append(lineJson["choices"][0]["messages"][0]["delta"])
                        yield format_as_ndjson(response)
                    elif role == "assistant": 
                        if response['apim-request-id'] and DEBUG_LOGGING: 
                            logging.debug(f"RESPONSE apim-request-id: {response['apim-request-id']}")
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                        yield format_as_ndjson(response)
                    else:
                        deltaText = lineJson["choices"][0]["messages"][0]["delta"]["content"]
                        if deltaText != "[DONE]":
                            response["choices"][0]["messages"].append({
                                "role": "assistant",
                                "content": deltaText
                            })
                            yield format_as_ndjson(response)
    except Exception as e:
        yield format_as_ndjson({"error" + str(e)})

def formatApiResponseNoStreaming(rawResponse):
    if 'error' in rawResponse:
        return {"error": rawResponse["error"]}
    response = {
        "id": rawResponse["id"],
        "model": rawResponse["model"],
        "created": rawResponse["created"],
        "object": rawResponse["object"],
        "choices": [{
            "messages": []
        }],
    }
    toolMessage = {
        "role": "tool",
        "content": rawResponse["choices"][0]["message"]["context"]["messages"][0]["content"]
    }
    assistantMessage = {
        "role": "assistant",
        "content": rawResponse["choices"][0]["message"]["content"]
    }
    response["choices"][0]["messages"].append(toolMessage)
    response["choices"][0]["messages"].append(assistantMessage)

    return response

def formatApiResponseStreaming(rawResponse):
    if 'error' in rawResponse:
        return {"error": rawResponse["error"]}
    response = {
        "id": rawResponse["id"],
        "model": rawResponse["model"],
        "created": rawResponse["created"],
        "object": rawResponse["object"],
        "choices": [{
            "messages": []
        }],
    }

    if rawResponse["choices"][0]["delta"].get("context"):
        messageObj = {
            "delta": {
                "role": "tool",
                "content": rawResponse["choices"][0]["delta"]["context"]["messages"][0]["content"]
            }
        }
        response["choices"][0]["messages"].append(messageObj)
    elif rawResponse["choices"][0]["delta"].get("role"):
        messageObj = {
            "delta": {
                "role": "assistant",
            }
        }
        response["choices"][0]["messages"].append(messageObj)
    else:
        if rawResponse["choices"][0]["end_turn"]:
            messageObj = {
                "delta": {
                    "content": "[DONE]",
                }
            }
            response["choices"][0]["messages"].append(messageObj)
        else:
            messageObj = {
                "delta": {
                    "content": rawResponse["choices"][0]["delta"]["content"],
                }
            }
            response["choices"][0]["messages"].append(messageObj)

    return response

def conversation_with_data(request_body):
    print("conversation_with_data")
    body, headers = prepare_body_headers_with_data(request)
    base_url = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    endpoint = f"{base_url}openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"
    history_metadata = request_body.get("history_metadata", {})

    if not SHOULD_STREAM:
        r = requests.post(endpoint, headers=headers, json=body)
        status_code = r.status_code
        r = r.json()
        if AZURE_OPENAI_PREVIEW_API_VERSION == "2023-06-01-preview":
            r['history_metadata'] = history_metadata
            return Response(format_as_ndjson(r), status=status_code)
        else:
            result = formatApiResponseNoStreaming(r)
            result['history_metadata'] = history_metadata
            return Response(format_as_ndjson(result), status=status_code)

    else:
        return Response(stream_with_data(body, headers, endpoint, history_metadata), mimetype='text/event-stream')

def stream_without_data(response, history_metadata={}):
    #print("stream_without_data")
    responseText = ""
    for line in response:
        if line["choices"]:
            deltaText = line["choices"][0]["delta"].get('content')
        else:
            deltaText = ""
        #print(deltaText)
        if deltaText and deltaText != "[DONE]":
            responseText = deltaText

        response_obj = {
            "id": line["id"],
            "model": line["model"],
            "created": line["created"],
            "object": line["object"],
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": responseText
                }]
            }],
            "history_metadata": history_metadata
        }
        #print(response_obj)
        #print('--------')
        yield format_as_ndjson(response_obj)

def our_stream_without_data(full_response, history_metadata={}):
    print("our_stream_without_data: sending response text")
    print(full_response)

    # words = []

    # for c in full_response:
    #     words.append(c)

    # Split the text into words
    words = re.findall(r'\S+|\n|\W', full_response)

    print(words)

    session_uuid = uuid.uuid4()
    session_created_time = int(time.time())

    # Iterate through the words
    start_message = ["#START#", "#START-SECOND#"]
    words = start_message + words

    print(words)
    for word in words:
        response_obj = {}

        if word == "#START#":
            response_obj = {
                "id": "",
                "model": "",
                "created": 0,
                "object": "",
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": ""
                    }]
                }],
                "history_metadata": {}
            }
        elif word == "#START-SECOND#":
            response_obj = {
                "id": "chatcmpl-" + str(session_uuid),
                "model": "gpt-35-turbo",
                "created": session_created_time,
                "object": "chat.completion.chunk",
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": ""
                    }]
                }],
                "history_metadata": {}
            }
        elif word == '':
            response_obj = {
                "id": "chatcmpl-" + str(session_uuid),
                "model": "gpt-35-turbo",
                "created": session_created_time,
                "object": "chat.completion.chunk",
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": ' '
                    }]
                }],
                "history_metadata": history_metadata
            }
        else:
            response_obj = {
                "id": "chatcmpl-" + str(session_uuid),
                "model": "gpt-35-turbo",
                "created": session_created_time,
                "object": "chat.completion.chunk",
                "choices": [{
                    "messages": [{
                        "role": "assistant",
                        "content": word
                    }]
                }],
                "history_metadata": history_metadata
            }

        yield format_as_ndjson(response_obj)

    # for line in response:
    #     print(line)
    #     if line["choices"]:
    #         deltaText = line["choices"][0]["delta"].get('content')
    #     else:
    #         deltaText = ""
    #     if deltaText and deltaText != "[DONE]":
    #         responseText = deltaText

    #     response_obj = {
    #         "id": line["id"],
    #         "model": line["model"],
    #         "created": line["created"],
    #         "object": line["object"],
    #         "choices": [{
    #             "messages": [{
    #                 "role": "assistant",
    #                 "content": responseText
    #             }]
    #         }],
    #         "history_metadata": history_metadata
    #     }
    #     yield format_as_ndjson(response_obj)


def extract_content(input_string):
    start_substring = "KQL_START"
    end_substring = "KQL_END"

    start_index = input_string.find(start_substring)
    end_index = input_string.find(end_substring)

    if start_index != -1 and end_index != -1 and start_index < end_index:
        result = input_string[start_index + len(start_substring):end_index].strip()
        return result
    else:
        return None

def conversation_without_data(request_body):
    print("conversation_without_data")
    print(request_body)

    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    openai.api_version = "2023-08-01-preview"
    openai.api_key = AZURE_OPENAI_KEY

    request_messages = request_body["messages"]
    print(request_messages)
    messages = [{
        "role": "system",
        "content": "Behaviour: Behave like an Azure Data Explorer Kusto Query Language expert. Your task is to provide KQL statements to the various question being asked for the below tables.\nContextual Data: I have a dataset wherein i have a table named 'enriched-edr'. It has the below comma separated columns and the corresponding datatype for each column in the format column_name:datatype. Schema for the various tables is as below:\nTable #1: 'enriched-edr'\n\nColumns:\n\neventTimeFlow:datetime,flowRecord_flowRecordType:string,flowRecord_subscriberInfo_imsi:string,flowRecord_keys_sessionId:real,flowRecord_dpiStringInfo_application:string,flowRecord_dpiStringInfo_layer7Protocol:string,flowRecord_gatewayInfo_gwNodeID:string,flowRecord_dpiStringInfo_operatingSystem:string,flowRecord_creationtime_timesecs:string,flowRecord_creationtime_timeusecs:string,flowRecord_networkStatsInfo_downlinkFlowPeakThroughput:long,flowRecord_networkStatsInfo_uplinkFlowPeakThroughput:long,flowRecord_networkStatsInfo_downlinkFlowActivityDuration:string,flowRecord_networkStatsInfo_uplinkFlowActivityDuration:string,flowRecord_networkStatsInfo_downlinkInitialRTT_timesecs:long,flowRecord_networkStatsInfo_downlinkInitialRTT_timeusecs:long,flowRecord_networkStatsInfo_uplinkInitialRTT_timesecs:long,flowRecord_networkStatsInfo_uplinkInitialRTT_timeusecs:long,flowRecord_networkStatsInfo_closureReason:string,flowRecord_networkPerfInfo_initialRTT_timesecs:long,flowRecord_networkPerfInfo_initialRTT_timeusecs:long,flowRecord_networkPerfInfo_HttpTtfbTime_timesecs:long,flowRecord_networkPerfInfo_HttpTtfbTime_timeusecs:long,flowRecord_dataStats_upLinkOctets:long,flowRecord_dataStats_downLinkOctets:long,flowRecord_dataStats_downLinkPackets:long,flowRecord_dataStats_downLinkDropPackets:long,flowRecord_dataStats_upLinkPackets:long,flowRecord_dataStats_upLinkDropPackets:long,flowRecord_tcpRetransInfo_downlinkRetransBytes:long,flowRecord_tcpRetransInfo_uplinkRetransBytes:long,flowRecord_tcpRetransInfo_downlinkRetransPackets:string,flowRecord_tcpRetransInfo_uplinkRetransPackets:string,flowRecord_ipTuple_networkIpAddress:string,flowRecord_ipTuple_networkPort:long,flowRecord_ipTuple_protocol:string,flowRecord_keys_flowId:real,eventTimeSession:datetime,sessionRecord_subscriberInfo_userLocationInfo_ecgi_eci:long,sessionRecord_keys_sessionId:real,sessionRecord_subscriberInfo_imsi:real,sessionRecord_subscriberInfo_msisdn:string,sessionRecord_subscriberInfo_imeisv_tac:longsessionRecord_servingNetworkInfo_apnId:string,sessionRecord_servingNetworkInfo_nodeAddress:string,sessionRecord_servingNetworkInfo_nodePlmnId_mcc:longsessionRecord_servingNetworkInfo_nodePlmnId_mnc:long,eventTimeHTTP:string,httpRecord_keys_flowId:real,httpRecord_keys_sessionId:real,httpRecord_httpTransactionRecord_subscriberInformation_gwNodeID:string,httpRecord_httpTransactionRecord_subscriberInformation_imsi:real,httpRecord_keys_transactionId:long,httpRecord_httpTransactionRecord_dpiInformation_application:string,httpRecord_httpTransactionRecord_subscriberInformation_realApn:string,httpRecord_httpTransactionRecord_requestInformation_failureReason:string,httpRecord_httpTransactionRecord_responseInformation_responseStatus:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_maxDownstreamMSS:long,httpRecord_httpTransactionRecord_tcpServerConnectionInformation_MSS:long,httpRecord_httpTransactionRecord_tcpServerConnectionInformation_rtt:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelNoneTime:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelMildTime:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelModerateTime:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_congestionLevelSevereTime:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_minRTT:long,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_maxRTT:string,httpRecord_httpTransactionRecord_tcpClientConnectionInformation_avgRTT:string,httpRecord_httpTransactionRecord_tcpSplicingInformation_tlsSNI:string,httpRecord_httpTransactionRecord_udpProxyInformation_quicSni:string,httpRecord_httpTransactionRecord_requestInformation_serverUrl_Host:string,Column71:string\nInstructions: In your response, write an KQL query based on the user input message.\n1. Answer in a concise manner.\n2. Answer only with the KQL query statement\n3. Do not add any extra text in your response.\n4. Do not preface your response with anything.\n5. In the section which has KQL statement, prefix the KQL statement with \"KQL_START\" and suffix the section with \"KQL_END\"\n6. Don't use the prefix and suffix in case the response is expressive and does not contain only KQL statement and also has regular english sentences"}
    ]

    last_message = request_messages[-1]
    if last_message:
        messages.append({
            "role": last_message["role"] ,
            "content": last_message["content"]
        })
    print("Last Message:")
    print(last_message)

    # for message in request_messages:
    #     if message:
    #         messages.append({
    #             "role": message["role"] ,
    #             "content": message["content"]
    #         })
    
    # latest_message = request_body['messages'][-1]['content']
    # print(latest_message)
    # df = call_openai_kql_response(latest_message)
    # print(df)
    # df_as_text = df.to_string()
    # print("--------")
    # print(df_as_text)

    print("Messages:")
    print(messages)

    response = openai.ChatCompletion.create(
        engine=utils.OPENAI_DEPLOYMENT_NAME,
        messages = messages,
        temperature=float(AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
        top_p=float(AZURE_OPENAI_TOP_P),
        stop=AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        stream=SHOULD_STREAM
    )

    history_metadata = request_body.get("history_metadata", {})

    if not SHOULD_STREAM:
        print("mohit 1")
        response_obj = {
            "id": response,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }]
            }],
            "history_metadata": history_metadata
        }

        return jsonify(response_obj), 200
    else:
        #print("Response Object:", response)

        # Extract the sentence from the response
        words = []
        for chunk in response:
            #print(chunk)
            token = None
            if chunk["choices"]:
                token = chunk["choices"][-1]["delta"].get('content')
            else:
                token = ""
            
            if token and token != "[DONE]":
                #print("'" + token + "'")
                words.append(token)

        sentence = ''.join(words)

        print("Sentence:")
        print(sentence)

        # Execute Query if KQL
        if "KQL_START" in sentence:
            kql_query = extract_content(sentence)

            if (kql_query is None):
                response_to_stream_back = sentence
            else:
                print("KQL Query: " + kql_query)

                # Surround the table name with [''] if not already there!
                if "['enriched-edr']" not in kql_query:
                    kql_query = kql_query.replace("enriched-edr", "['enriched-edr']", 1)

                print("KQL Query after formatting table name: " + kql_query)

                try:
                    kql_response = client.execute("aoienriched", kql_query)

                    print("KQL Response:")
                    print(kql_response)

                    # Convert KQL Response to a Pandas DataFrame
                    df = dataframe_from_result_table(kql_response.primary_results[0])

                    # Convert DataFrame as text
                    df_as_text = df.to_string()

                    print("Dataframe as text:")
                    print(df_as_text)

                    response_to_stream_back = "Query: " + kql_query + "\n\n" + df_as_text
                except Exception as e:
                    response_to_stream_back = f"Query: {kql_query}\n\nError while executing this query.\n\nError: {e}"
        else:
            print ("Regular Sentence: ")
            # Stream the original OpenAI response back as a stream
            response_to_stream_back = sentence

        # Stream the text back to Frontend
        return Response(our_stream_without_data(response_to_stream_back, history_metadata), mimetype='text/event-stream')

@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    request_body = request.json
    return conversation_internal(request_body)

def conversation_internal(request_body):
    try:
        use_data = should_use_data()
        if use_data:
            return conversation_with_data(request_body)
        else:
            return conversation_without_data(request_body)
    except Exception as e:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(e)}), 500

## Conversation History API ## 
@app.route("/history/generate", methods=["POST"])
def add_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = generate_title(request.json["messages"])
            conversation_dict = cosmos_conversation_client.create_conversation(user_id=user_id, title=title)
            conversation_id = conversation_dict['id']
            history_metadata['title'] = title
            history_metadata['date'] = conversation_dict['createdAt']
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request.json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "user":
            cosmos_conversation_client.create_message(
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
        else:
            raise Exception("No user message found")
        
        # Submit request to Chat Completions for response
        request_body = request.json
        history_metadata['conversation_id'] = conversation_id
        request_body['history_metadata'] = history_metadata
        return conversation_internal(request_body)
       
    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@app.route("/history/update", methods=["POST"])
def update_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not cosmos_conversation_client:
            raise Exception("CosmosDB is not configured")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")
            
        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request.json["messages"]
        if len(messages) > 0 and messages[-1]['role'] == "assistant":
            if len(messages) > 1 and messages[-2].get('role', None) == "tool":
                # write the tool message first
                cosmos_conversation_client.create_message(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2]
                )
            # write the assistant message
            cosmos_conversation_client.create_message(
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1]
            )
        else:
            raise Exception("No bot messages found")
        
        # Submit request to Chat Completions for response
        response = {'success': True}
        return jsonify(response), 200
       
    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500

@app.route("/history/delete", methods=["DELETE"])
def delete_conversation():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    try: 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        ## delete the conversation messages from cosmos first
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        ## Now delete the conversation 
        deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation_id)

        return jsonify({"message": "Successfully deleted conversation and messages", "conversation_id": conversation_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500

@app.route("/history/list", methods=["GET"])
def list_conversations():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## get the conversations from cosmos
    conversations = cosmos_conversation_client.get_conversations(user_id)
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200

@app.route("/history/read", methods=["POST"])
def get_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## get the conversation object and the related messages from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404
    
    # get the messages for the conversation from cosmos
    conversation_messages = cosmos_conversation_client.get_messages(user_id, conversation_id)

    ## format the messages in the bot frontend format
    messages = [{'id': msg['id'], 'role': msg['role'], 'content': msg['content'], 'createdAt': msg['createdAt']} for msg in conversation_messages]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200

@app.route("/history/rename", methods=["POST"])
def rename_conversation():
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    
    ## get the conversation from cosmos
    conversation = cosmos_conversation_client.get_conversation(user_id, conversation_id)
    if not conversation:
        return jsonify({"error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."}), 404

    ## update the title
    title = request.json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation['title'] = title
    updated_conversation = cosmos_conversation_client.upsert_conversation(conversation)

    return jsonify(updated_conversation), 200

@app.route("/history/delete_all", methods=["DELETE"])
def delete_all_conversations():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']

    # get conversations for user
    try:
        conversations = cosmos_conversation_client.get_conversations(user_id)
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404
        
        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = cosmos_conversation_client.delete_messages(conversation['id'], user_id)

            ## Now delete the conversation 
            deleted_conversation = cosmos_conversation_client.delete_conversation(user_id, conversation['id'])

        return jsonify({"message": f"Successfully deleted conversation and messages for user {user_id}"}), 200
    
    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500
    

@app.route("/history/clear", methods=["POST"])
def clear_messages():
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user['user_principal_id']
    
    ## check request for conversation_id
    conversation_id = request.json.get("conversation_id", None)
    try: 
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        
        ## delete the conversation messages from cosmos
        deleted_messages = cosmos_conversation_client.delete_messages(conversation_id, user_id)

        return jsonify({"message": "Successfully deleted messages in conversation", "conversation_id": conversation_id}), 200
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500

@app.route("/history/ensure", methods=["GET"])
def ensure_cosmos():
    if not AZURE_COSMOSDB_ACCOUNT:
        return jsonify({"error": "CosmosDB is not configured"}), 404
    
    if not cosmos_conversation_client or not cosmos_conversation_client.ensure():
        return jsonify({"error": "CosmosDB is not working"}), 500

    return jsonify({"message": "CosmosDB is configured and working"}), 200


def generate_title(conversation_messages):
    ## make sure the messages are sorted by _ts descending
    title_prompt = 'Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Respond with a json object in the format {{"title": string}}. Do not include any other commentary or description.'

    messages = [{'role': msg['role'], 'content': msg['content']} for msg in conversation_messages]
    messages.append({'role': 'user', 'content': title_prompt})

    try:
        ## Submit prompt to Chat Completions for response
        base_url = AZURE_OPENAI_ENDPOINT if AZURE_OPENAI_ENDPOINT else f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
        openai.api_type = "azure"
        openai.api_base = base_url
        openai.api_version = "2023-03-15-preview"
        openai.api_key = AZURE_OPENAI_KEY
        completion = openai.ChatCompletion.create(    
            engine=utils.OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=1,
            max_tokens=64 
        )
        title = json.loads(completion['choices'][0]['message']['content'])['title']
        return title
    except Exception as e:
        return messages[-2]['content']

if __name__ == "__main__":
    app.run()