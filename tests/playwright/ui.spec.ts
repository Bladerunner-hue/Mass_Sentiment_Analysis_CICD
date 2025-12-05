import { test, expect, Page } from '@playwright/test';

const TEST_USER = {
  username: process.env.PLAYWRIGHT_USERNAME || 'playwright_user',
  email: process.env.PLAYWRIGHT_EMAIL || 'playwright_user@example.com',
  password: process.env.PLAYWRIGHT_PASSWORD || 'P@ssw0rd123!',
};

async function loginOrRegister(page: Page) {
  await page.goto('/');

  const signInLink = page.getByRole('link', { name: /sign in/i });
  if (await signInLink.isVisible()) {
    await signInLink.click();
    await page.getByLabel(/email address/i).fill(TEST_USER.email);
    await page.getByLabel(/password/i).fill(TEST_USER.password);
    await page.getByRole('button', { name: /sign in/i }).click();

    // If login failed (still on login page), try registration flow
    if (await page.getByRole('button', { name: /sign in/i }).isVisible()) {
      await page.getByRole('link', { name: /create a new account/i }).click();
      await page.getByLabel(/username/i).fill(TEST_USER.username);
      await page.getByLabel(/email address/i).fill(TEST_USER.email);
      await page.getByLabel(/password/i).fill(TEST_USER.password);
      await page.getByLabel(/confirm password/i).fill(TEST_USER.password);
      await page.getByRole('button', { name: /create account/i }).click();
    }
  }

  // Assert logged in (nav should show Dashboard link)
  await expect(page.getByRole('link', { name: /dashboard/i })).toBeVisible();
}

test.describe('Sentiment Analyzer UI smoke tests', () => {
  test('landing page renders navigation and hero content', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText(/Sentiment Analyzer/i)).toBeVisible();
    await expect(page.getByRole('link', { name: /analyze/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /datasets/i })).toBeVisible();
  });

  test('analyze flow submits text and shows results', async ({ page }) => {
    await loginOrRegister(page);
    await page.getByRole('link', { name: /analyze/i }).click();
    await page.getByLabel(/text to analyze/i).fill('Playwright UI test positive experience');
    await page.getByRole('button', { name: /analyze/i }).click();

    // Results section appears
    await expect(page.locator('#results')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/Sentiment/i)).toBeVisible();
  });

  test('batch processing page shows upload controls', async ({ page }) => {
    await loginOrRegister(page);
    await page.getByRole('link', { name: /batch processing/i }).click();
    await expect(page.getByText(/Upload CSV/i)).toBeVisible();
    await expect(page.locator('input[type="file"]')).toBeVisible();
  });

  test('datasets page renders sections for sources', async ({ page }) => {
    await loginOrRegister(page);
    await page.getByRole('link', { name: /datasets/i }).click();
    await expect(page.getByText(/Kaggle/i)).toBeVisible();
    await expect(page.getByText(/HuggingFace/i)).toBeVisible();
    await expect(page.getByText(/Twitter/i)).toBeVisible();
  });
});
