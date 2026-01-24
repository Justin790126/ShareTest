#ifndef VIEWPATHPICKERWIDGET_H
#define VIEWPATHPICKERWIDGET_H

#include <QWidget>
#include <QString>

class QLineEdit;
class QToolButton;

class ViewPathPickerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewPathPickerWidget(QWidget* parent = 0);

    QString path() const;
    void setPath(const QString& p);

    void setDialogTitle(const QString& title);
    void setNameFilter(const QString& filter);
    void setPlaceholder(const QString& text);

    void setReadOnly(bool ro);
    void setBrowseEnabled(bool en);

signals:
    void pathChanged(const QString& path);

private slots:
    void browse();
    void onTextEdited(const QString& text);

private:
    QLineEdit*   m_edit;
    QToolButton* m_btn;

    QString m_dialogTitle;
    QString m_nameFilter;
};

#endif
