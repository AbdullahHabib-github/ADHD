import '/flutter_flow/flutter_flow_util.dart';
import 'register_widget.dart' show RegisterWidget;
import 'package:flutter/material.dart';

class RegisterModel extends FlutterFlowModel<RegisterWidget> {
  ///  State fields for stateful widgets in this page.

  // State field(s) for fullName widget.
  FocusNode? fullNameFocusNode;
  TextEditingController? fullNameController;
  String? Function(BuildContext, String?)? fullNameControllerValidator;
  // State field(s) for email widget.
  FocusNode? emailFocusNode;
  TextEditingController? emailController;
  String? Function(BuildContext, String?)? emailControllerValidator;
  // State field(s) for pass1 widget.
  FocusNode? pass1FocusNode;
  TextEditingController? pass1Controller;
  late bool pass1Visibility;
  String? Function(BuildContext, String?)? pass1ControllerValidator;
  // State field(s) for pass2 widget.
  FocusNode? pass2FocusNode;
  TextEditingController? pass2Controller;
  late bool pass2Visibility;
  String? Function(BuildContext, String?)? pass2ControllerValidator;

  /// Initialization and disposal methods.

  @override
  void initState(BuildContext context) {
    pass1Visibility = false;
    pass2Visibility = false;
  }

  @override
  void dispose() {
    fullNameFocusNode?.dispose();
    fullNameController?.dispose();

    emailFocusNode?.dispose();
    emailController?.dispose();

    pass1FocusNode?.dispose();
    pass1Controller?.dispose();

    pass2FocusNode?.dispose();
    pass2Controller?.dispose();
  }

  /// Action blocks are added here.

  /// Additional helper methods are added here.
}
