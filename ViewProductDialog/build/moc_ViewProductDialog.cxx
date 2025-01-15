/****************************************************************************
** Meta object code from reading C++ file 'ViewProductDialog.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.7)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../ViewProductDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ViewProductDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.7. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ViewAddProductDialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_ViewAddProductDialog[] = {
    "ViewAddProductDialog\0"
};

void ViewAddProductDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData ViewAddProductDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ViewAddProductDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_ViewAddProductDialog,
      qt_meta_data_ViewAddProductDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ViewAddProductDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ViewAddProductDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ViewAddProductDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewAddProductDialog))
        return static_cast<void*>(const_cast< ViewAddProductDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int ViewAddProductDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_ViewProductDialog[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   18,   18,   18, 0x05,
      32,   18,   18,   18, 0x05,
      45,   18,   18,   18, 0x05,
      61,   18,   18,   18, 0x05,
      81,   77,   18,   18, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_ViewProductDialog[] = {
    "ViewProductDialog\0\0loadConfig()\0"
    "saveConfig()\0addNewProduct()\0"
    "delSelProduct()\0key\0searchKeyChanged(QString)\0"
};

void ViewProductDialog::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ViewProductDialog *_t = static_cast<ViewProductDialog *>(_o);
        switch (_id) {
        case 0: _t->loadConfig(); break;
        case 1: _t->saveConfig(); break;
        case 2: _t->addNewProduct(); break;
        case 3: _t->delSelProduct(); break;
        case 4: _t->searchKeyChanged((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData ViewProductDialog::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ViewProductDialog::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_ViewProductDialog,
      qt_meta_data_ViewProductDialog, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ViewProductDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ViewProductDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ViewProductDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewProductDialog))
        return static_cast<void*>(const_cast< ViewProductDialog*>(this));
    return QDialog::qt_metacast(_clname);
}

int ViewProductDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 5)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void ViewProductDialog::loadConfig()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}

// SIGNAL 1
void ViewProductDialog::saveConfig()
{
    QMetaObject::activate(this, &staticMetaObject, 1, 0);
}

// SIGNAL 2
void ViewProductDialog::addNewProduct()
{
    QMetaObject::activate(this, &staticMetaObject, 2, 0);
}

// SIGNAL 3
void ViewProductDialog::delSelProduct()
{
    QMetaObject::activate(this, &staticMetaObject, 3, 0);
}

// SIGNAL 4
void ViewProductDialog::searchKeyChanged(const QString & _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}
QT_END_MOC_NAMESPACE
