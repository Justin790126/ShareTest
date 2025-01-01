#ifndef VIEW_LYR_DIALOG_H
#define VIEW_LYR_DIALOG_H

#include <QDialog>
#include <QtGui>

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
    QComboBox *cbbOutlineWidth = NULL;
    QSpinBox *spbOutlineWidth = NULL;

    QPushButton *btnOK = NULL;
    QPushButton *btnCancel = NULL;
    QPushButton *btnApply = NULL;
};

#endif // VIEW_LYR_DIALOG_H