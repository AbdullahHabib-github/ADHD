import 'dart:async';

import 'package:collection/collection.dart';

import '/backend/schema/util/firestore_util.dart';
import '/backend/schema/util/schema_util.dart';

import 'index.dart';

class StudybuddyRecord extends FirestoreRecord {
  StudybuddyRecord._(
    super.reference,
    super.data,
  ) {
    _initializeFields();
  }

  // "location" field.
  LatLng? _location;
  LatLng? get location => _location;
  bool hasLocation() => _location != null;

  // "name" field.
  String? _name;
  String get name => _name ?? '';
  bool hasName() => _name != null;

  void _initializeFields() {
    _location = snapshotData['location'] as LatLng?;
    _name = snapshotData['name'] as String?;
  }

  static CollectionReference get collection =>
      FirebaseFirestore.instance.collection('studybuddy');

  static Stream<StudybuddyRecord> getDocument(DocumentReference ref) =>
      ref.snapshots().map((s) => StudybuddyRecord.fromSnapshot(s));

  static Future<StudybuddyRecord> getDocumentOnce(DocumentReference ref) =>
      ref.get().then((s) => StudybuddyRecord.fromSnapshot(s));

  static StudybuddyRecord fromSnapshot(DocumentSnapshot snapshot) =>
      StudybuddyRecord._(
        snapshot.reference,
        mapFromFirestore(snapshot.data() as Map<String, dynamic>),
      );

  static StudybuddyRecord getDocumentFromData(
    Map<String, dynamic> data,
    DocumentReference reference,
  ) =>
      StudybuddyRecord._(reference, mapFromFirestore(data));

  @override
  String toString() =>
      'StudybuddyRecord(reference: ${reference.path}, data: $snapshotData)';

  @override
  int get hashCode => reference.path.hashCode;

  @override
  bool operator ==(other) =>
      other is StudybuddyRecord &&
      reference.path.hashCode == other.reference.path.hashCode;
}

Map<String, dynamic> createStudybuddyRecordData({
  LatLng? location,
  String? name,
}) {
  final firestoreData = mapToFirestore(
    <String, dynamic>{
      'location': location,
      'name': name,
    }.withoutNulls,
  );

  return firestoreData;
}

class StudybuddyRecordDocumentEquality implements Equality<StudybuddyRecord> {
  const StudybuddyRecordDocumentEquality();

  @override
  bool equals(StudybuddyRecord? e1, StudybuddyRecord? e2) {
    return e1?.location == e2?.location && e1?.name == e2?.name;
  }

  @override
  int hash(StudybuddyRecord? e) =>
      const ListEquality().hash([e?.location, e?.name]);

  @override
  bool isValidKey(Object? o) => o is StudybuddyRecord;
}
