import 'package:flutter/material.dart';

class FFAppState extends ChangeNotifier {
  static FFAppState _instance = FFAppState._internal();

  factory FFAppState() {
    return _instance;
  }

  FFAppState._internal();

  static void reset() {
    _instance = FFAppState._internal();
  }

  Future initializePersistedState() async {}

  void update(VoidCallback callback) {
    callback();
    notifyListeners();
  }

  String _threadId = '';
  String get threadId => _threadId;
  set threadId(String value) {
    _threadId = value;
  }

  dynamic _FirstResponse;
  dynamic get FirstResponse => _FirstResponse;
  set FirstResponse(dynamic value) {
    _FirstResponse = value;
  }
}
