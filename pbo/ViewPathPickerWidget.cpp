#include "ViewPathPickerWidget.h"

#include <QLineEdit>
#include <QToolButton>
#include <QHBoxLayout>
#include <QFileDialog>

ViewPathPickerWidget::ViewPathPickerWidget(QWidget* parent)
    : QWidget(parent)
    , m_edit(0)
    , m_btn(0)
    , m_dialogTitle("Select file")
    , m_nameFilter("All Files (*)")
{
    QHBoxLayout* h = new QHBoxLayout(this);
    h->setContentsMargins(4, 2, 4, 2);
    h->setSpacing(6);

    m_edit = new QLineEdit(this);
    m_edit->setPlaceholderText("type path or browse...");

    m_btn = new QToolButton(this);
    m_btn->setText("...");
    m_btn->setToolTip("Browse");
    m_btn->setFixedWidth(26);

    h->addWidget(m_edit, 1);
    h->addWidget(m_btn, 0);

    connect(m_btn, SIGNAL(clicked()), this, SLOT(browse()));
    connect(m_edit, SIGNAL(textEdited(QString)), this, SLOT(onTextEdited(QString)));
}

QString ViewPathPickerWidget::path() const
{
    return m_edit->text();
}

void ViewPathPickerWidget::setPath(const QString& p)
{
    if (m_edit->text() == p) return;
    m_edit->setText(p);
    emit pathChanged(m_edit->text());
}

void ViewPathPickerWidget::setDialogTitle(const QString& title)
{
    m_dialogTitle = title;
}

void ViewPathPickerWidget::setNameFilter(const QString& filter)
{
    m_nameFilter = filter;
}

void ViewPathPickerWidget::setPlaceholder(const QString& text)
{
    m_edit->setPlaceholderText(text);
}

void ViewPathPickerWidget::setReadOnly(bool ro)
{
    m_edit->setReadOnly(ro);
}

void ViewPathPickerWidget::setBrowseEnabled(bool en)
{
    m_btn->setEnabled(en);
}

void ViewPathPickerWidget::browse()
{
    if (!m_btn->isEnabled()) return;

    QString startDir;
    const QString current = m_edit->text();
    if (!current.isEmpty())
        startDir = current;

    QString p = QFileDialog::getOpenFileName(this, m_dialogTitle, startDir, m_nameFilter);
    if (p.isEmpty()) return;

    m_edit->setText(p);
    emit pathChanged(p);
}

void ViewPathPickerWidget::onTextEdited(const QString& text)
{
    emit pathChanged(text);
}
