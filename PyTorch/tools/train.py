#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# Time    :   2022/3/29
# Author  :   XavierYorke
# Contact :   mzlxavier1230@gmail.com


import pandas as pd
import os
import torch
from PyTorch.models import dice_cal


def train_step(model, images, labels):
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()
    # 正向传播求损失
    outputs, output_h3 = model(images)
    loss = model.loss_func(outputs, output_h3, labels)
    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()
    dice = dice_cal(outputs, labels)
    return loss.item(), dice.item()


def valid_step(model, images, labels):
    # 预测模式，dropout层不发生作用
    model.eval()
    outputs, output_h3 = model(images)
    loss = model.loss_func(outputs, output_h3, labels)
    dice = dice_cal(outputs, labels)
    return loss.item(), dice.item()


def train(model, start, epochs, dl_train, dl_valid, log_step_freq, save_step_freq, logger, device, output_dir):
    notes = pd.DataFrame(columns=["epoch", "loss", "val_loss", "dice", "val_dice"])
    logger.info("Start Training...")
    for epoch in range(start, epochs + 1):
        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        avd = 0.0
        step = 1
        for step, batch in enumerate(dl_train, 1):
            images = batch['image']
            labels = batch['label']
            images = images.to(device)
            labels = labels.to(device)

            loss, dice = train_step(model, images, labels)
            # 打印batch级别日志
            loss_sum += loss
            avd += dice
            if step % log_step_freq == 0:
                msg = "[step = {}] loss: {:.3f} dice: {:.3f}".format(step, loss_sum / step, avd / step)
                logger.info(msg)

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_avd = 0.0
        val_step = 1
        for val_step, batch in enumerate(dl_valid, 1):
            images = batch['image']
            labels = batch['label']
            images = images.to(device)
            labels = labels.to(device)

            val_loss, val_dice = valid_step(model, images, labels)
            val_loss_sum += val_loss
            val_avd += val_dice

            if step % log_step_freq == 0:
                msg = "[val step = {}] loss: {:.3f} dice: {:.3f}".format(val_step, val_loss_sum / val_step, val_avd / val_step)
                logger.info(msg)

        # save model
        if epoch % save_step_freq == 0:
            save_pth = os.path.join(output_dir, 'epoch-' + str(epoch) + '.pth')
            state = model.state_dict()
            torch.save(state, save_pth)

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, val_loss_sum / val_step, avd / step,  val_avd / val_step)
        notes.loc[epoch - 1] = info

        # 打印epoch级别日志
        msg = ("[EPOCH = {}], loss = {:.3f}, val_loss = {:.3f}, dice = {:.3f}, val_dice = {:.3f}".format(*info))
        logger.info(msg)
        notes.to_csv(os.path.join(output_dir, 'notes.csv'), index=False)
