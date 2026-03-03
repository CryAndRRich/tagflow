import copy
from typing import List
import torch
from torch.amp import autocast, GradScaler

def train_model_stage_1(model: torch.nn.Module, 
                        train_loader: torch.utils.data.DataLoader, 
                        val_loader: torch.utils.data.DataLoader, 
                        loss_fn: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler, 
                        attribute_idx: int, 
                        num_epochs: int, 
                        early_stopping: int,
                        device: torch.device,
                        verbose: float = False) -> torch.nn.Module:
    """
    Huấn luyện mô hình cho một thuộc tính cụ thể (giai đoạn 1)
    
    Tham số:
        model: Mô hình cần huấn luyện
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        loss_fn: Hàm mất mát
        optimizer: Bộ tối ưu hóa
        scheduler: Bộ điều chỉnh tốc độ học 
        attribute_idx: Chỉ số của thuộc tính cần huấn luyện
        num_epochs: Số epoch tối đa để huấn luyện
        early_stopping: Số epoch không cải thiện tối đa trước khi dừng
        device: Thiết bị huấn luyện
        verbose: Nếu True, in thông tin huấn luyện sau mỗi epoch
    
    Trả về:
        torch.nn.Module: Mô hình đã được huấn luyện tốt nhất
    """
    model = model.to(device)
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    scaler = GradScaler()

    early_stopping_count = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(batch_x)
                loss = loss_fn(outputs[0], batch_y[:, attribute_idx]).mean()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs[0].detach(), dim=1)
            train_correct += (preds == batch_y[:, attribute_idx]).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total
            
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                with autocast():
                    outputs = model(batch_x)
                preds = torch.argmax(outputs[0], dim=1)
                val_correct += (preds == batch_y[:, attribute_idx]).sum().item()
                val_total += batch_y.size(0)
                
        val_acc = val_correct / val_total
        
        if verbose:
            print(f"Attribute {attribute_idx + 1} Training | "
                f"Epoch {epoch + 1:02d}/{num_epochs} | "
                f"Train/Val Acc: {train_acc:.4f} / {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if best_acc == 1.0 or early_stopping_count == early_stopping:
            break
            
    model.load_state_dict(best_weights)
    if verbose:
        print(f"   => Hoàn thành Attribute {attribute_idx + 1} | Best Val Acc: {best_acc:.4f} \n")
    
    return model


def train_model_stage_2(model: torch.nn.Module, 
                        train_loader: torch.utils.data.DataLoader, 
                        val_loader: torch.utils.data.DataLoader, 
                        loss_fns: List[torch.nn.Module], 
                        optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler, 
                        attribute_list: List[int],
                        num_epochs: int, 
                        alpha_max_loss: float, 
                        early_stopping: int,
                        checkpoint_file: str,
                        device: torch.device,
                        verbose: float = False) -> None:
    """
    Huấn luyện mô hình cho tất cả các thuộc tính cùng lúc (giai đoạn 2)
    
    Tham số:
        model: Mô hình cần huấn luyện
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        loss_fns: Danh sách các hàm mất mát cho từng thuộc tính
        optimizer: Bộ tối ưu hóa
        scheduler: Bộ điều chỉnh tốc độ học
        attribute_list: Danh sách chỉ số của các thuộc tính
        num_epochs: Số epoch tối đa để huấn luyện
        alpha_max_loss: Hệ số điều chỉnh cho phần mất mát lớn nhất
        early_stopping: Số epoch không cải thiện tối đa trước khi dừng
        checkpoint_file: Đường dẫn để lưu mô hình tốt nhất
        device: Thiết bị huấn luyện
        verbose: Nếu True, in thông tin huấn luyện sau mỗi epoch
    """
    model = model.to(device)
    best_exact_match = 0.0
    scaler = GradScaler()
    
    early_stopping_count = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_exact_match_correct = 0
        train_total_samples = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(batch_x)
                
                attr_losses = []
                for i in range(len(attribute_list)):
                    attr_losses.append(loss_fns[i](outputs[i], batch_y[:, i]))
                
                stacked_losses = torch.stack(attr_losses, dim=1) 
                base_loss = torch.sum(stacked_losses, dim=1)     
                worst_loss, _ = torch.max(stacked_losses, dim=1) 
                
                sample_losses = base_loss + (alpha_max_loss * worst_loss)
                loss = torch.mean(sample_losses)
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            preds = [torch.argmax(out.detach(), dim=1) for out in outputs]
            preds_matrix = torch.stack(preds, dim=1)
            correct_rows = torch.all(preds_matrix == batch_y, dim=1)
            
            train_exact_match_correct += correct_rows.sum().item()
            train_total_samples += batch_y.size(0)

            scheduler.step()
            
        avg_train_loss = train_loss / len(train_loader)
        train_exact_match = train_exact_match_correct / train_total_samples
        
        model.eval()
        val_loss = 0.0
        val_exact_match_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                with autocast():
                    outputs = model(batch_x)
                    
                    attr_losses = []
                    for i in range(len(attribute_list)):
                        attr_losses.append(loss_fns[i](outputs[i], batch_y[:, i]))
                    
                    stacked_losses = torch.stack(attr_losses, dim=1)
                    base_loss = torch.sum(stacked_losses, dim=1)
                    worst_loss, _ = torch.max(stacked_losses, dim=1)
                    
                    sample_losses = base_loss + (alpha_max_loss * worst_loss)
                    batch_loss = torch.mean(sample_losses)
                    
                val_loss += batch_loss.item()
                
                preds = [torch.argmax(out, dim=1) for out in outputs]
                preds_matrix = torch.stack(preds, dim=1)
                correct_rows = torch.all(preds_matrix == batch_y, dim=1)
                val_exact_match_correct += correct_rows.sum().item()
                val_total_samples += batch_y.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_exact_match = val_exact_match_correct / val_total_samples
        
        current_lr = optimizer.param_groups[0]["lr"]
        
        if verbose:
            print(f"Epoch {epoch + 1:02d}/{num_epochs} | "
                f"LR: {current_lr:.6f} | "
                f"Train/Val Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                f"Train/Val EM: {train_exact_match:.4f} / {val_exact_match:.4f}")
        
        if val_exact_match > best_exact_match:
            best_exact_match = val_exact_match
            torch.save(model.state_dict(), checkpoint_file)
            if verbose:
                print(f"  => Mô hình tốt nhất! (EM: {best_exact_match:.4f})")
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            if verbose:
                print(f"  => Mô hình không cải thiện ({early_stopping_count}/{early_stopping})")

        if best_exact_match == 1.0 or early_stopping_count == early_stopping:
            if verbose:
                print("  => Dừng mô hình")
            break