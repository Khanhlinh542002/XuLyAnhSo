package Pages;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

public class TrangLienHeHK {
    private WebDriver driver;
    private WebDriverWait wait;

    public TrangLienHeHK(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10);
    }

    // Các locators hiện có
    private By btnDangNhap = By.id("OpenIdConnect");
    private By byEmail = By.name("loginfmt");
    private By byPass = By.name("passwd");
    private By btnNext = By.id("idSIButton9");
    private By btnYes = By.xpath("/html/body/div/form/div/div/div[2]/div[1]/div/div/div/div/div/div[3]/div/div[2]/div/div[3]/div[2]/div/div/div[2]/input");

    // Thêm học kì
    private By termButton = By.xpath("/html/body/div[2]/div[1]/div[2]/ul/li[2]/a");
    private By btnThemHocKi = By.xpath("/html/body/div[2]/div[2]/div[3]/div/section/div/div/div/div[2]/div/div/div[1]/div[2]/div/div[2]/button");
    private By inputHocKi = By.id("id");
    private By dropdownNamBatDau = By.id("select2-start_year-container");
    private By dropdownNamKetThuc = By.id("select2-end_year-container");
    private By btnLuu = By.xpath("/html/body/div[3]/div[2]/form/div[7]/button[2]");

    // Xóa học kì
    private By btnXoaHocKi = By.xpath("/html/body/div[2]/div[2]/div[3]/div/section/div/div/div/div[2]/div/div/table/tbody/tr[1]/td[9]/a[2]/i");
    private By btnXacNhanXoa = By.xpath("/html/body/div[7]/div/div[6]/button[1]");
    private By btnHuyXoa = By.xpath("/html/body/div[7]/div/div[6]/button[2]");
    
    // Các locators mới từ test case XoaHocKi
    private By hkButton = By.xpath("/html/body/div[2]/div[2]/div[3]/div/section/div/div/div/div[2]/ul/li[2]/a");
    private By timKiemInput = By.xpath("/html/body/div[2]/div[2]/div[3]/div/section/div/div/div/div[2]/div/div/div[1]/div[2]/div/div[1]/div/label/input");
    private By thongBaoXoa = By.id("swal2-html-container");

    public void login(String email, String password) {
        driver.findElement(btnDangNhap).click();
        driver.switchTo().window((String) driver.getWindowHandles().toArray()[1]);
        driver.findElement(byEmail).sendKeys(email);
        driver.findElement(btnNext).click();
        driver.findElement(byPass).sendKeys(password);
        driver.findElement(btnNext).click();
        driver.findElement(btnYes).click();
    }
    
    public void addHocKi(String hocKi, String namBatDau, String namKetThuc) {
        wait.until(ExpectedConditions.elementToBeClickable(termButton)).click();
        wait.until(ExpectedConditions.elementToBeClickable(btnThemHocKi)).click();
        driver.findElement(inputHocKi).sendKeys(hocKi);
        driver.findElement(dropdownNamBatDau).click();
        driver.findElement(By.xpath("//li[text()='" + namBatDau + "']")).click();
        driver.findElement(dropdownNamKetThuc).click();
        driver.findElement(By.xpath("//li[text()='" + namKetThuc + "']")).click();
        driver.findElement(btnLuu).click();
    }

    // Các phương thức mới từ test case XoaHocKi
    public void clickTermButton() {
        wait.until(ExpectedConditions.elementToBeClickable(termButton)).click();
    }
    
    public void clickHocKiButton() {
        wait.until(ExpectedConditions.elementToBeClickable(hkButton)).click();
    }
    
    public void timKiemHocKi(String hocKi) {
        wait.until(ExpectedConditions.elementToBeClickable(timKiemInput)).sendKeys(hocKi);
    }
    
    public void clickXoaHocKi() {
        wait.until(ExpectedConditions.elementToBeClickable(btnXoaHocKi)).click();
    }
    
    public void xacNhanXoa() {
        wait.until(ExpectedConditions.elementToBeClickable(btnXacNhanXoa)).click();
    }
    
    public void huyXoa() {
        wait.until(ExpectedConditions.elementToBeClickable(btnHuyXoa)).click();
    }
    
    public String getThongBaoXoa() {
        return wait.until(ExpectedConditions.visibilityOfElementLocated(thongBaoXoa)).getText();
    }
    
    public boolean isHocKiTonTai() {
        return !driver.findElements(btnXoaHocKi).isEmpty();
    }
}
