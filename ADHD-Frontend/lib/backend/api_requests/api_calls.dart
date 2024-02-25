import 'dart:convert';

import '/flutter_flow/flutter_flow_util.dart';
import 'api_manager.dart';

export 'api_manager.dart' show ApiCallResponse;

const _kPrivateApiFunctionName = 'ffPrivateApiCall';

class UploadBookCall {
  static Future<ApiCallResponse> call({
    FFUploadedFile? pdfFile,
  }) async {
    return ApiManager.instance.makeApiCall(
      callName: 'Upload book',
      apiUrl: 'https://3khsmtgg-5000.inc1.devtunnels.ms/upload_book',
      callType: ApiCallType.POST,
      headers: {},
      params: {
        'pdf_file': pdfFile,
      },
      bodyType: BodyType.MULTIPART,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      alwaysAllowBody: false,
    );
  }
}

class ChatbotCall {
  static Future<ApiCallResponse> call({
    String? prompt = '',
    String? threadId = '',
  }) async {
    final ffApiRequestBody = '''
{
  "thread_id": "$threadId",
  "prompt": "$prompt"
}''';
    return ApiManager.instance.makeApiCall(
      callName: 'Chatbot',
      apiUrl: 'https://3khsmtgg-5000.inc1.devtunnels.ms/chatbot',
      callType: ApiCallType.POST,
      headers: {},
      params: {},
      body: ffApiRequestBody,
      bodyType: BodyType.JSON,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      alwaysAllowBody: false,
    );
  }
}

class ChatbotgetCall {
  static Future<ApiCallResponse> call() async {
    return ApiManager.instance.makeApiCall(
      callName: 'chatbotget',
      apiUrl: 'https://3khsmtgg-5000.inc1.devtunnels.ms/chatbot',
      callType: ApiCallType.GET,
      headers: {},
      params: {},
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      alwaysAllowBody: false,
    );
  }

  static String? response(dynamic response) => castToType<String>(getJsonField(
        response,
        r'''$.Response''',
      ));
  static String? threadid(dynamic response) => castToType<String>(getJsonField(
        response,
        r'''$.Thread_id''',
      ));
}

class ApiPagingParams {
  int nextPageNumber = 0;
  int numItems = 0;
  dynamic lastResponse;

  ApiPagingParams({
    required this.nextPageNumber,
    required this.numItems,
    required this.lastResponse,
  });

  @override
  String toString() =>
      'PagingParams(nextPageNumber: $nextPageNumber, numItems: $numItems, lastResponse: $lastResponse,)';
}

String _serializeList(List? list) {
  list ??= <String>[];
  try {
    return json.encode(list);
  } catch (_) {
    return '[]';
  }
}

String _serializeJson(dynamic jsonVar, [bool isList = false]) {
  jsonVar ??= (isList ? [] : {});
  try {
    return json.encode(jsonVar);
  } catch (_) {
    return isList ? '[]' : '{}';
  }
}
