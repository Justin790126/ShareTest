/****************************************************************************
** Meta object code from reading C++ file 'lcOvlProduct.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../lcOvlProduct.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'lcOvlProduct.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_lcOvlProduct[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x08,
      33,   13,   13,   13, 0x08,
      55,   13,   13,   13, 0x08,
      77,   13,   13,   13, 0x08,
      99,   13,   13,   13, 0x08,
     125,  121,   13,   13, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_lcOvlProduct[] = {
    "lcOvlProduct\0\0handelRcsvReaded()\0"
    "handleAddNewProduct()\0handleLoadOvlConfig()\0"
    "handleSaveOvlConfig()\0handleDelSelProduct()\0"
    "key\0handleSearchKeyChanged(QString)\0"
};

void lcOvlProduct::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        lcOvlProduct *_t = static_cast<lcOvlProduct *>(_o);
        switch (_id) {
        case 0: _t->handelRcsvReaded(); break;
        case 1: _t->handleAddNewProduct(); break;
        case 2: _t->handleLoadOvlConfig(); break;
        case 3: _t->handleSaveOvlConfig(); break;
        case 4: _t->handleDelSelProduct(); break;
        case 5: _t->handleSearchKeyChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData lcOvlProduct::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject lcOvlProduct::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_lcOvlProduct,
      qt_meta_data_lcOvlProduct, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &lcOvlProduct::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *lcOvlProduct::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *lcOvlProduct::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_lcOvlProduct))
        return static_cast<void*>(const_cast< lcOvlProduct*>(this));
    return QObject::qt_metacast(_clname);
}

int lcOvlProduct::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
