#ifndef VIEW_LYR_DIALOG_H
#define VIEW_LYR_DIALOG_H

#include <QDialog>
#include <QtGui>

#pragma once

#include<QComboBox>
#include<QLabel>
#include<QLineEdit>
#include<QListWidget>

class QPenWidth : public QLabel
{
    Q_OBJECT

public:
    QPenWidth(QWidget  *parent = NULL);
    ~QPenWidth();

    void SetDrawParameters(int drawType, int isize);
protected:
    void paintEvent(QPaintEvent *) override;
private:
    int m_drawType;//绘制类型(画点/画线)
    int m_size;//大小
};


class  QPenWidget : public QLineEdit
{
    Q_OBJECT
public:
    QPenWidget(QWidget *parent = NULL);
    ~QPenWidget();

    void updatePen(const int& index, const int& type);
protected:
    void mousePressEvent(QMouseEvent *event);

signals:
    void click(const int& index);
private:
    QLabel* m_pLabel;
    QPenWidth* m_pCssLabel;
    int m_index;
    int m_type;
};

class  QLineWidthCombobox :public QComboBox
{
    Q_OBJECT
public:
    QLineWidthCombobox(QWidget *parent = NULL);
    ~QLineWidthCombobox();

    void SetType(int drawType);

    int GetCurrentIndex();
    void SetCurrentIndex(int index);

    void SetList(QList<int>&list);
    QList<int> GetList();

private slots:
    void onClickPenWidget(const int& index);

signals:
    void SelectedItemChanged(int);

public:
    void appendItem(const int& index);
private:
    QPenWidget* m_pPenEdit;
    QListWidget* m_pListWidget;
    QList<QColor> m_colorsList;
    int m_drawType;
    int m_index;
    QList<int> m_list;
};

class ViewLyrDialog : public QDialog
{
    Q_OBJECT
public:
    explicit ViewLyrDialog(QWidget *parent = nullptr);
    ~ViewLyrDialog() = default;

    QLineEdit *GetLyrNameLineEdit() { return leLyrName; }

    QLineEdit *GetLyrNumLineEdit() { return leLyrNum; }
    QLineEdit *GetLyrDTypeLineEdit() { return leLyrDType; }
    QPushButton *GetFillMoreColorsBtn() { return btnFillMoreColors; }
    QLineEdit *GetFillMoreColorsHexLineEdit() { return leFillColorHex; }
    QSlider *GetFillColorAlphaSlider() { return sldFillColorAlpha; }
    QPushButton *GetOutlineMoreColorsBtn() { return btnOutlineMoreColors; }
    QLineEdit *GetOutlineColorHexLineEdit() { return leOutlineColorHex; }
    QSlider *GetOutlineColorAlphaSlider() { return sldOutlineColorAlpha; }
    QComboBox *GetOutlineWidthComboBox() { return cbbOutlineWidth; }
    QSpinBox *GetOutlineColorSpinBox() { return spbOutlineWidth; }
    QPushButton *GetOKBtn() { return btnOK; }
    QPushButton *GetCancelBtn() { return btnCancel; }
    QPushButton *GetApplyBtn() { return btnApply; }

private:
    QFrame *CreateSeparator();
    void Widgets();
    void Layout();

    QLineEdit *leLyrName = NULL;
    QLineEdit *leLyrNum = NULL;
    QLineEdit *leLyrDType = NULL;
    QPushButton *btnFillMoreColors = NULL;
    QLineEdit *leFillColorHex = NULL;
    QSlider *sldFillColorAlpha = NULL;
    QLabel* lblFillAlphaValue = NULL;

    QPushButton *btnOutlineMoreColors = NULL;
    QLineEdit *leOutlineColorHex = NULL;
    QSlider *sldOutlineColorAlpha = NULL;
    QLabel* lblOutlineAlphaValue = NULL;
    QLineWidthCombobox *cbbOutlineWidth = NULL;
    QSpinBox *spbOutlineWidth = NULL;

    QPushButton *btnOK = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnApply = NULL;
};

#endif // VIEW_LYR_DIALOG_H