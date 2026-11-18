import { test, expect } from '@playwright/test';

test('Has note', async ({ page }) => {
    await page.goto('https://demo.playwright.dev/todomvc/#/');

    await page.getByRole('textbox').fill('купить хлеб');

    await page.getByRole('textbox').press('Enter');

    await expect(page.getByText('купить хлеб')).toBeVisible();
});