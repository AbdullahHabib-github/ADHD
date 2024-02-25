import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';

Future initFirebase() async {
  if (kIsWeb) {
    await Firebase.initializeApp(
        options: const FirebaseOptions(
            apiKey: "AIzaSyCZMBFSHiz8Cd14v3wwg8BT_nQPpx1k0Qo",
            authDomain: "adhd-415110.firebaseapp.com",
            projectId: "adhd-415110",
            storageBucket: "adhd-415110.appspot.com",
            messagingSenderId: "533258550191",
            appId: "1:533258550191:web:51076fe2cc27f4a5ff4d5a",
            measurementId: "G-EZE1TMWD6E"));
  } else {
    await Firebase.initializeApp();
  }
}
