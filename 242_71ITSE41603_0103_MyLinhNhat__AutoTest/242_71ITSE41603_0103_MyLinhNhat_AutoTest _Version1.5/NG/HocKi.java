package NG;

import Pages.TrangLienHeHK;
import io.github.bonigarcia.wdm.WebDriverManager;
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.testng.Assert;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class HocKi {
    private WebDriver driver;
    private WebDriverWait wait;
    private TrangLienHeHK trangLienHe;
    private String baseUrl = "https://cntttest.vanlanguni.edu.vn:18081/Phancong02/Account/Login";

    @BeforeTest
    public void StartWeb() {
        try {
            WebDriverManager.chromedriver().setup();
            ChromeOptions options = new ChromeOptions();
            options.addArguments("--remote-allow-origins=*");
            options.addArguments("--disable-notifications");

            driver = new ChromeDriver(options);
            driver.manage().window().maximize();
            wait = new WebDriverWait(driver, 10);

            driver.get(baseUrl);

            // Skip security warning if it appears
            try {
                wait.until(ExpectedConditions.elementToBeClickable(By.id("details-button"))).click();
                wait.until(ExpectedConditions.elementToBeClickable(By.id("proceed-link"))).click();
            } catch (Exception e) {
                System.out.println("Không cần bỏ qua cảnh báo bảo mật.");
            }

            wait.until(ExpectedConditions.elementToBeClickable(By.xpath("/html/body/div/div[3]/div[2]/div/div/div/div/form/div/div/div/button"))).click();

            String originalWindow = driver.getWindowHandle();
            for (String windowHandle : driver.getWindowHandles()) {
                if (!windowHandle.equals(originalWindow)) {
                    driver.switchTo().window(windowHandle);
                    break;
                }
            }

            wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("loginfmt")))
                .sendKeys("nhat.2274802010606@vanlanguni.vn");
            wait.until(ExpectedConditions.elementToBeClickable(By.id("idSIButton9"))).click();

            wait.until(ExpectedConditions.visibilityOfElementLocated(By.name("passwd")))
                .sendKeys("*Pensi0793");
            wait.until(ExpectedConditions.elementToBeClickable(By.id("idSIButton9"))).click();

            // Wait for additional login steps
            wait.until(ExpectedConditions.elementToBeClickable(By.id("idSIButton9"))).click();

            driver.switchTo().window(originalWindow);
            wait.until(ExpectedConditions.urlContains("/Phancong02/"));
            
            // Initialize TrangLienHeHK instance
            trangLienHe = new TrangLienHeHK(driver);

        } catch (Exception e) {
            System.err.println("Lỗi trong StartWeb: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }

    @Test(priority = 1)
    public void addHocKiTest() {
        trangLienHe.addHocKi("109", "2024", "2025");
    }
    
    @Test(priority = 2)
    public void TrongButton() {
        trangLienHe.clickTermButton();
        WebElement btnThemHocKi = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("/html/body/div[2]/div[2]/div[3]/div/section/div/div/div/div[2]/div/div/div[1]/div[2]/div/div[2]/button")));
        btnThemHocKi.click();

        WebElement btnLuu = wait.until(ExpectedConditions.elementToBeClickable(By.xpath("/html/body/div[3]/div[2]/form/div[5]/button[2]")));
        btnLuu.click();
        
        WebElement inputhocki = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("id-error")));
        String actualMessage = inputhocki.getText();
        String expectedMessage = "Bạn chưa nhập học kỳ";
        Assert.assertEquals(actualMessage, expectedMessage, "Message mismatch: ");
    }

    @Test(priority = 3)
    public void XoaHocKiThanhCong() {
        trangLienHe.clickTermButton();
        trangLienHe.clickHocKiButton();
        trangLienHe.timKiemHocKi("109");
        trangLienHe.clickXoaHocKi();
        trangLienHe.xacNhanXoa();
    }
    
    @Test(priority = 4)
    public void Xoahockithatbai() {
        trangLienHe.clickTermButton();
        trangLienHe.clickHocKiButton();
        trangLienHe.clickXoaHocKi();
        
        String actualMessage = trangLienHe.getThongBaoXoa();
        String expectedMessage = "Bạn có chắc muốn xoá học kì này?";
        Assert.assertEquals(actualMessage, expectedMessage, "Message mismatch on delete confirmation: ");
        
        trangLienHe.xacNhanXoa();
        
        String actualMessage1 = trangLienHe.getThongBaoXoa();
        String expectedMessage1 = "Không thể xoá do học kỳ này đã có dữ liệu!";
        Assert.assertEquals(actualMessage1, expectedMessage1, "Message mismatch on delete failure: ");
        
        System.out.println(trangLienHe.isHocKiTonTai() ? "Xóa học kì thất bại" : "Xóa học kì thành công");
        
        trangLienHe.xacNhanXoa();
    }
    
    @Test(priority = 5)
    public void HuyXoaNganh() {
        trangLienHe.clickTermButton();
        trangLienHe.clickHocKiButton();
        trangLienHe.clickXoaHocKi();
        
        String actualMessage = trangLienHe.getThongBaoXoa();
        String expectedMessage = "Bạn có chắc muốn xoá học kì này?";
        Assert.assertEquals(actualMessage, expectedMessage, "Message mismatch on cancel delete: ");
        
        trangLienHe.huyXoa();
    }

    @AfterTest
    public void tearDown() {
        if (driver != null) {
            driver.quit();
        }
    }
}
